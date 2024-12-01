import json
import re

def extract_vision_neurons(weight_data):
    """
    提取与 vision_model 相关的神经元，保持 `.weight` 和 `.bias` 完整键，同时支持 `.bias` 层偏移。
    """
    vision_neurons = {}
    for key, value in weight_data.items():
        if key.startswith("vision_model"):  # 提取 vision_model 的键
            # 检查是否是 `.bias`，并尝试偏移层
            if ".bias" in key:
                adjusted_key = shift_bias_key(key)
                if adjusted_key not in vision_neurons:
                    vision_neurons[adjusted_key] = []
                vision_neurons[adjusted_key].extend(value)
            else:
                if key not in vision_neurons:
                    vision_neurons[key] = []
                vision_neurons[key].extend(value)
        elif isinstance(value, dict):  # 递归处理嵌套字典
            nested_neurons = extract_vision_neurons(value)
            vision_neurons.update(nested_neurons)
    return vision_neurons

def shift_bias_key(key):
    """
    偏移 `.bias` 的键到上一层。
    例如，将 `vision_model.encoder.layers.1.self_attn.q_proj.bias` 映射到 `vision_model.encoder.layers.0.self_attn.q_proj`。
    """
    if ".bias" not in key:
        return key
    # 匹配层索引并向前移动一层
    match = re.search(r"(layers\.)(\d+)", key)
    if match:
        layer_prefix, layer_index = match.groups()
        shifted_index = int(layer_index) - 1
        shifted_key = key.replace(f"{layer_prefix}{layer_index}", f"{layer_prefix}{shifted_index}")
        return shifted_key.replace(".bias", "")  # 移除 `.bias`
    return key.replace(".bias", "")  # 无法匹配则直接移除 `.bias`

# 加载 JSON 文件
with open('dormant_neurons_activation.json', 'r') as f:
    activation_data = json.load(f)

with open('dormant_neurons_weight.json', 'r') as f:
    weight_data = json.load(f)

# 提取权重相关的 vision_model 神经元
vision_weight_data = extract_vision_neurons(weight_data)

# 对比 activation 和 weight 数据
overlap_results = {}
for layer, neurons in activation_data.items():
    # 匹配 `.weight` 和（偏移后的）`.bias`
    weight_neurons = vision_weight_data.get(f"{layer}.weight", [])
    bias_neurons = vision_weight_data.get(layer, [])  # 已偏移的 `.bias`
    
    # 转换为集合，便于操作
    activation_set = set(neurons)
    weight_set = set(weight_neurons)
    bias_set = set(bias_neurons)
    
    # 分别计算交集和差集
    weight_intersection = activation_set & weight_set
    weight_union = activation_set | weight_set
    weight_only = weight_set - activation_set
    activation_only_weight = activation_set - weight_set

    bias_intersection = activation_set & bias_set
    bias_union = activation_set | bias_set
    bias_only = bias_set - activation_set
    activation_only_bias = activation_set - bias_set

    # 计算重叠率
    weight_overlap_ratio = len(weight_intersection) / len(weight_union) * 100 if weight_union else 0
    bias_overlap_ratio = len(bias_intersection) / len(bias_union) * 100 if bias_union else 0

    # 保存结果
    overlap_results[layer] = {
        "weight": {
            "activation_only": list(activation_only_weight),
            "weight_only": list(weight_only),
            "shared": list(weight_intersection),
            "union_overlap_ratio": weight_overlap_ratio,
        },
        "bias": {
            "activation_only": list(activation_only_bias),
            "bias_only": list(bias_only),
            "shared": list(bias_intersection),
            "union_overlap_ratio": bias_overlap_ratio,
        },
    }

# 输出结果
for layer, result in overlap_results.items():
    print(f"Layer: {layer}")
    print(f"  Weight:")
    #print(f"    Activation Only: {result['weight']['activation_only']}")
    #print(f"    Weight Only: {result['weight']['weight_only']}")
    #print(f"    Shared: {result['weight']['shared']}")
    print(f"    Union Overlap Ratio: {result['weight']['union_overlap_ratio']:.2f}%")
    print(f"  Bias:")
    #print(f"    Activation Only: {result['bias']['activation_only']}")
    #print(f"    Bias Only: {result['bias']['bias_only']}")
    #print(f"    Shared: {result['bias']['shared']}")
    print(f"    Union Overlap Ratio: {result['bias']['union_overlap_ratio']:.2f}%")
    print()
