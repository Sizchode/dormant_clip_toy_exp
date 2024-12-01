from transformers import CLIPModel, CLIPProcessor
from tracker import CLIPDormantNeuronTracker
from trainer import Trainer
from dataloader import GTSRBTrainDataset, GTSRBTestDataset
import yaml
from kaiming_reinit import VisionModelInitializer

from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import os
import json
import random
import numpy as np
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
config["learning_rate"] = float(config["learning_rate"])
config["weight_decay"] = float(config["weight_decay"])
config["label_smoothing"] = float(config["label_smoothing"])
config["batch_size"] = int(config["batch_size"])
config["gradient_accumulation_steps"] = int(config["gradient_accumulation_steps"])
config["num_epochs"] = int(config["num_epochs"])
set_seed(config["seed"])

# Prepare the environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up data loaders
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = GTSRBTrainDataset(root_dir=config["train_data_path"], transform=transform)
test_dataset = GTSRBTestDataset(
    annotations_file=config["val_annotations_file"], img_dir=config["val_img_dir"], transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Load CLIP model and precompute text features
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
initializer = VisionModelInitializer()
initializer.reinitialize_model(model)
initializer.verify_initialization(model)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_inputs = processor(text=[f"traffic sign {i}" for i in range(43)], padding=True, return_tensors="pt").to(device)
with torch.no_grad():
    text_features = model.get_text_features(input_ids=text_inputs["input_ids"]).detach()

# First Stage: Train the model
print("Starting first-stage fine-tuning...")
tracker = CLIPDormantNeuronTracker(model=model, threshold=0.01)
tracker.register_activation_hooks()  # Track activations during the first stage
tracker.initialize_total_neurons()

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    text_features=text_features,
    config=config,
    device=device,
    tracker=tracker,
    tracking_mode="activation"  # Focus on activation tracking
)

# Run training
trainer.train_activation()

# Save dormant neurons after the first stage
save_path = "dormant_neurons_activation.json"
tracker.save(save_path, mode="activation")
print(f"First-stage dormant neurons saved to {save_path}")


# Second-Stage Fine-Tuning
print("Starting second-stage fine-tuning...")
with open("dormant_neurons_activation.json", "r") as f:
    dormant_neurons = json.load(f)

# Load model for second stage
model_2nd_stage = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
initializer.reinitialize_model(model_2nd_stage)
initializer.verify_initialization(model_2nd_stage)

# Freeze all parameters first
for param in model_2nd_stage.parameters():
    param.requires_grad = False

# Unfreeze dormant neurons only
for name, module in model_2nd_stage.named_modules():
    if name in dormant_neurons:  # Unfreeze specific neurons by name
        indices = dormant_neurons[name]
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.requires_grad = True
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.requires_grad = True

# Second-stage tracker and trainer
tracker_2nd_stage = CLIPDormantNeuronTracker(model=model_2nd_stage, threshold=0.01)
tracker_2nd_stage.register_activation_hooks()  # Track activations during second stage

trainer_2nd_stage = Trainer(
    model=model_2nd_stage,
    train_loader=train_loader,
    test_loader=test_loader,
    text_features=text_features,
    config=config,
    device=device,
    tracker=tracker_2nd_stage,
    tracking_mode="activation"  # Second stage focuses on activation tracking
)

# Run second-stage training
trainer_2nd_stage.train_activation()

