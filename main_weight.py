from transformers import CLIPModel, CLIPProcessor
from tracker import DormancyTracker
from kaiming_reinit import VisionModelInitializer
from trainer import Trainer
from dataloader import GTSRBTrainDataset, GTSRBTestDataset
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import json
import random 
import numpy as np

# Load configuration
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
tracker = DormancyTracker(model=model, threshold=0.01)

# Initialize total neurons for weight tracking
tracker.initialize_total_neurons()

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    text_features=text_features,
    config=config,
    device=device,
    tracker=tracker,
    # tracking_mode="weight"  # Change to weight tracking
)

# Train the model
print("Starting first-stage fine-tuning...")
trainer.train_weight()

# Track weight updates and calculate dormant neurons
tracker.track_weight_updates()
weight_dormant_ratio = tracker.calculate_dormant_ratio()
print(f"Final Dormant Neuron Ratio (Weight): {weight_dormant_ratio:.4f}")

# Save dormant neurons
tracker.save("dormant_neurons_weight.json")

# Load model for second stage
model_2nd_stage = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
initializer.reinitialize_model(model_2nd_stage)
initializer.verify_initialization(model_2nd_stage)

# Second-Stage Fine-Tuning
print("Starting second-stage fine-tuning...")
tracker.load("dormant_neurons_weight.json")
for name, param in model_2nd_stage.vision_model.named_parameters():
    if name not in tracker.dormant_neurons:
        param.requires_grad = False

# Second-stage tracker and trainer
# tracker_2nd_stage = DormancyTracker(model=model_2nd_stage, threshold=0.01)

trainer_2nd_stage = Trainer(
    model=model_2nd_stage,
    train_loader=train_loader,
    test_loader=test_loader,
    text_features=text_features,
    config=config,
    device=device,
    # tracker=tracker_2nd_stage,
    # tracking_mode="weight"
)

# Run second-stage training
trainer_2nd_stage.train_weight()

# Calculate and save dormant ratios after second stage
#weight_ratio_2nd_stage = tracker_2nd_stage.calculate_dormant_ratio(mode="weight_update")
#print(f"Second-Stage Weight Dormant Ratio: {weight_ratio_2nd_stage:.4f}")
#tracker_2nd_stage.save("second_stage_dormant_neurons_weight.json", mode="weight_update")
