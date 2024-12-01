import torch
from torch import optim
import torch.nn as nn
from tqdm import tqdm

def tuner(model, train_loader, test_loader, text_features, config, device, tracker=None):
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    for epoch in range(config["num_epochs"]):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            logits = model.get_image_features(images) @ text_features.T
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save weights for tracking
        if tracker:
            tracker.save_weights()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model.get_image_features(images) @ text_features.T
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        print(f"Epoch {epoch + 1} Validation Accuracy: {correct / total:.2%}")
