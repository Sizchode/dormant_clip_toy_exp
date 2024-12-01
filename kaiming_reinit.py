import torch.nn as nn
import torch.nn.init as init

class VisionModelInitializer:
    """
    Class to handle the initialization and verification of weights for CLIP Vision Model.
    """

    def __init__(self):
        self.initialized_modules = set()

    def init_weights(self, module, module_name=""):
        """
        Initialize the weights of a module and record initialized modules.
        """
        initialized = False  # Flag to track if the module is initialized

        # Initialize different types of modules
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                init.constant_(module.bias, 0)
            initialized = True
        elif isinstance(module, nn.Linear):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                init.constant_(module.bias, 0)
            initialized = True
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0, std=0.02)
            initialized = True
        elif isinstance(module, nn.LayerNorm):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
            initialized = True
        elif isinstance(module, nn.MultiheadAttention):
            if hasattr(module, "in_proj_weight"):
                init.kaiming_normal_(module.in_proj_weight, mode="fan_out", nonlinearity="relu")
            if hasattr(module, "out_proj"):
                init.kaiming_normal_(module.out_proj.weight, mode="fan_out", nonlinearity="relu")
            initialized = True

        if initialized:
            self.initialized_modules.add(module)

    def reinitialize_model(self, model):
        """
        Reinitialize the weights of all modules in the vision model.
        """
        for name, module in model.vision_model.named_modules():
            self.init_weights(module, module_name=name)

    def verify_initialization(self, model):
        """
        Verify if all modules in the vision model have been successfully initialized.
        """
        uninitialized_modules = []

        for name, module in model.vision_model.named_modules():
            # Exclude modules without weights or biases
            if not any(hasattr(module, attr) for attr in ["weight", "bias"]):
                continue
            if module not in self.initialized_modules:
                uninitialized_modules.append(name)

        if not uninitialized_modules:
            print("\nYes, everything reinitialized!")
        else:
            print("\n[Verification] The following modules were NOT reinitialized:")
            for name in uninitialized_modules:
                print(f"  - Module: {name}")
