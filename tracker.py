import torch
import torch.nn as nn
import json
from collections import defaultdict

class CLIPDormantNeuronTracker:
    def __init__(self, model, threshold=0.01):
        """
        Initialize the tracker for dormant neurons.
        Args:
            model (nn.Module): CLIP model instance.
            threshold (float): Threshold to define dormant neurons (default: 0.01).
        """
        self.model = model
        self.threshold = threshold
        self.dormant_neurons = {"activation": {}}
        self.dormant_neurons_weight = defaultdict(list)
        self.total_neurons = {"activation": 0}
        self.total_neuron = 0
        self.prev_weights = None
        self.current_weights = None
    def clear_activation_data(self):
        """
        Clear all activation-based dormant neuron tracking data.
        """
        self.dormant_neurons["activation"].clear()
        self.total_neurons["activation"] = 0

    ### Activation-Based Tracking ###
    def initialize_total_neurons(self):
        """
        Compute and store the total number of neurons for all layers being tracked.
        """
        total_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm)):
                # Count neurons based on the number of output features (weight.shape[0])
                if hasattr(module, "weight") and module.weight is not None:
                    total_count += module.weight.shape[0]
        self.total_neurons["activation"] = total_count
        self.total_neuron = total_count
        print(f"Initialized Total Neurons: {total_count}")

    def register_activation_hooks(self):
        """
        Register hooks to capture activations across all layers, including embeddings.
        """
        for name, module in self.model.named_modules():
            # Ensure relevant modules like Linear, Conv2d, Embedding, and LayerNorm are covered
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm)):
                module.register_forward_hook(self._create_activation_hook(name))

    def _create_activation_hook(self, layer_name):
        """
        Internal method to create a hook for tracking activations.
        """
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            mean_activation = output.mean(dim=[0, 1])  # Batch and spatial dimensions
            dormant_indices = (mean_activation < self.threshold).nonzero(as_tuple=True)[0].tolist()
            self.dormant_neurons["activation"][layer_name] = dormant_indices
        return hook

    ### Unified Helper Functions ###
    def calculate_dormant_ratio(self, mode):
        """
        Calculate the ratio of dormant neurons based on the tracking mode.
        Args:
            mode (str): Either "activation" or "weight_update".
        """
        if mode not in self.dormant_neurons:
            raise ValueError(f"Invalid mode: {mode}. Choose 'activation' or 'weight_update'.")
        dormant_count = 0
        if mode == "weight_update":
            for name, indices in self.dormant_neurons_weight.items():
                # `indices` are the dormant neuron indices for the given layer
                dormant_count += len(indices)  # Count dormant neurons
                total_count = self.total_neuron
                print(f"Dormant Count: {dormant_count}, Total Neuron Count: {total_count}")
                return dormant_count / total_count if total_count > 0 else 0
        else:
            dormant_count = sum(len(indices) for indices in self.dormant_neurons[mode].values())
            total_count = self.total_neuron
            print(f"Dormant Count: {dormant_count}, Total Neuron Count: {total_count}")
            return dormant_count / total_count if total_count > 0 else 0

    def save(self, path, mode):
        """
        Save the tracked dormant neurons to a JSON file.
        Args:
            path (str): File path to save the JSON data.
            mode (str): Either "activation" or "weight_update".
        """
        if mode not in self.dormant_neurons:
            raise ValueError(f"Invalid mode: {mode}. Choose 'activation' or 'weight_update'.")

        with open(path, "w") as f:
            json.dump(self.dormant_neurons[mode], f, indent=4)

    def load(self, path, mode):
        """
        Load dormant neuron data from a JSON file.
        Args:
            path (str): File path to load the JSON data.
            mode (str): Either "activation" or "weight_update".
        """
        if mode not in self.dormant_neurons:
            raise ValueError(f"Invalid mode: {mode}. Choose 'activation' or 'weight_update'.")

        with open(path, "r") as f:
            self.dormant_neurons[mode] = json.load(f)

    ### Verification and Debugging ###
    def verify_all_hooks(self):
        """
        Verify if all relevant layers have hooks registered.
        """
        registered_layers = list(self.dormant_neurons["activation"].keys())
        print(f"Total layers with activation hooks: {len(registered_layers)}")
        print(f"Sample layers: {registered_layers[:10]}")

    def print_model_structure(self):
        """
        Print the model structure for debugging and verification purposes.
        """
        print("Listing all layers in the model:")
        for name, module in self.model.named_modules():
            print(f"{name}: {module}")


class DormancyTracker:
    def __init__(self, model, threshold=0.01):
        """
        Track dormant neurons by comparing weights between the last two epochs.
        """
        self.model = model
        self.threshold = threshold
        self.prev_weights = None
        self.current_weights = None
        self.dormant_neurons = defaultdict(list)
        self.total_neuron = 0

    def initialize_total_neurons(self):
        """
        Compute and store the total number of neurons for all layers being tracked.
        """
        total_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm)):
                # Count neurons based on the number of output features (weight.shape[0])
                if hasattr(module, "weight") and module.weight is not None:
                    total_count += module.weight.shape[0]
        self.total_neuron = total_count
        print(f"Initialized Total Neurons: {total_count}")


    def save_weights(self):
        """
        Save the current weights of the model for comparison.
        """
        self.prev_weights = self.current_weights
        self.current_weights = {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def track_weight_updates(self):
        """
        Compare weights from the last two saved states and identify dormant neurons.
        """
        if self.prev_weights is None or self.current_weights is None:
            raise ValueError("Weights for comparison are not available. Call save_weights() twice.")

        for name, curr_weight in self.current_weights.items():
            if name in self.prev_weights:
                diff = (curr_weight - self.prev_weights[name]).abs().mean(dim=0)
                dormant_indices = (diff < self.threshold).nonzero(as_tuple=True)[0].tolist()
                self.dormant_neurons[name] = dormant_indices

    def calculate_dormant_ratio(self):
        """
        Calculate the overall ratio of dormant neurons in the model.
        """
        dormant_count = 0
        total_neuron = self.total_neuron
        for name, indices in self.dormant_neurons.items():
            # Ensure the tensor has dimensions (not scalar)
            if self.current_weights[name].dim() > 0:
                #motal_neurons += self.current_weights[name].size(0)  # Total neurons in the layer
                dormant_count += len(indices)  # Number of dormant neurons in the layer

        if total_neuron == 0:
            return 0.0

        return dormant_count / total_neuron

    def save(self, path):
        """
        Save dormant neuron indices to a JSON file.
        """
        with open(path, "w") as f:
            json.dump(self.dormant_neurons, f, indent=4)

    def load(self, path):
        """
        Load dormant neuron indices from a JSON file.
        """
        with open(path, "r") as f:
            self.dormant_neurons = json.load(f)
