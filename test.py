from transformers import CLIPModel, CLIPProcessor
from kaiming_reinit import VisionModelInitializer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
initializer = VisionModelInitializer()
initializer.reinitialize_model(model)
initializer.verify_initialization(model)
