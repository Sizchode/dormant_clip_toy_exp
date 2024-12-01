import torch
from torch import nn, optim
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, test_loader, text_features, config, device, tracker=None, tracking_mode=None):
        """
        Initialize the trainer.

        Args:
        - model: The CLIP model to be trained.
        - train_loader: DataLoader for training.
        - test_loader: DataLoader for testing.
        - text_features: Precomputed text features.
        - config: Configuration dictionary.
        - device: Device to run the model on.
        - tracker: Tracker instance for dormant neurons.
        - tracking_mode: 'activation' or 'weight' to determine tracking method.
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.text_features = text_features
        self.config = config
        self.device = device
        self.tracker = tracker
        self.tracking_mode = tracking_mode


    def train_activation(self):
        """
        Train the model for multiple epochs, with dormant neuron tracking for the last epoch.
        """
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config["label_smoothing"])

        for epoch in range(self.config["num_epochs"]):
            self.model.train()
            total_loss = 0

            # Register hooks and clear data ONLY for the last epoch
            if self.tracker and self.tracking_mode == "activation" and epoch == self.config["num_epochs"] - 1:
                self.tracker.clear_activation_data()
                self.tracker.register_activation_hooks()
            
            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model.get_image_features(images) @ self.text_features.T
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1} Train Loss: {avg_loss:.4f}")

            # Validate the model after each epoch
            self.validate(criterion, epoch)

        # Calculate dormant neurons after the last epoch
        if self.tracker and self.tracking_mode == "activation":
            dormant_ratio = self.tracker.calculate_dormant_ratio(mode="activation")
            print(f"Final Dormant Neuron Ratio (Activation): {dormant_ratio:.4f}")
            self.tracker.save("dormant_neurons_activation.json", mode="activation")
        
    def train_weight(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config["label_smoothing"])

        for epoch in range(self.config["num_epochs"]):
            self.model.train()
            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model.get_image_features(images) @ self.text_features.T
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Save weights for tracking
            if self.tracker:
                self.tracker.save_weights()

            # Validation
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    logits = self.model.get_image_features(images) @ self.text_features.T
                    correct += (logits.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)
            print(f"Epoch {epoch + 1} Validation Accuracy: {correct / total:.2%}")
    def validate(self, criterion, epoch):
        """
        Validate the model on the test dataset.
        """
        self.model.eval()
        correct, total, val_loss = 0, 0, 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model.get_image_features(images) @ self.text_features.T

                val_loss += criterion(logits, labels).item()
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        avg_loss = val_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1} Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy
