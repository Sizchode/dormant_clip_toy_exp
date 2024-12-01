from transformers import CLIPModel
import json

# 加载 CLIP 模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# 加载 JSON 文件
json_path = "dormant_neurons_weight.json"  # 替换为实际 JSON 文件路径
with open(json_path, "r") as f:
    dormant_neurons = json.load(f)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 解冻 JSON 文件中指定的 dormant neurons
# Unfreeze only the dormant neurons specified in the JSON
for name, param in model.named_parameters():
    if name in dormant_neurons:  # Check if the parameter is in the JSON file
        param.requires_grad = True
        print(f"Unfroze: {name}")
    else:
        param.requires_grad = False  # Ensure all others remain frozen

# Verify the layers that are unfrozen
print("\nFinal Parameters with requires_grad=True:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)


# 验证冻结/解冻逻辑是否匹配 JSON 文件
model_parameters = {name for name, _ in model.named_parameters()}

# 检查 JSON 文件中的层是否匹配模型中的参数
unmatched_layers = [layer for layer in dormant_neurons.keys() if layer not in model_parameters]
if unmatched_layers:
    print("Unmatched layers from JSON:")
    for layer in unmatched_layers:
        print(layer)
else:
    print("All JSON layers matched with model parameters.")

# 验证解冻层是否真的影响梯度计算
print("\nParameters with requires_grad=True:")
requires_grad_layers = []
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
        requires_grad_layers.append(name)

# 检查 requires_grad=True 的参数是否包含 JSON 中的层
json_unmatched_requires_grad = [layer for layer in requires_grad_layers if layer not in dormant_neurons]
if json_unmatched_requires_grad:
    print("\nLayers with requires_grad=True but not in JSON:")
    for layer in json_unmatched_requires_grad:
        print(layer)
else:
    print("\nAll requires_grad=True parameters are from JSON.")

# 检查 JSON 中的层是否都被解冻
json_unmatched_frozen = [layer for layer in dormant_neurons if layer not in requires_grad_layers]
if json_unmatched_frozen:
    print("\nLayers in JSON but still frozen:")
    for layer in json_unmatched_frozen:
        print(layer)
else:
    print("\nAll JSON layers are properly unfrozen.")
