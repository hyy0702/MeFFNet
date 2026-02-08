import torch
import torch.nn as nn
from typing import List
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def cal_distance(m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    vector = m - n
    distance = torch.norm(vector, p=2, dim=1)
    distance = torch.clamp(distance, min=0.1)
    return distance


def cal_angle(a, b, c):
    ba = a - b
    bc = c - b

    dot = torch.matmul(ba.unsqueeze(-1).transpose(-2, -1), bc.unsqueeze(-1))
    cosine_angle = dot.squeeze(-1) / (torch.norm(ba, p=2, dim=1).reshape(-1, 1) * torch.norm(bc, p=2, dim=1).reshape(-1, 1))
    cosine_angle = torch.where(torch.logical_or(cosine_angle > 1, cosine_angle < -1), torch.round(cosine_angle), cosine_angle)
    angle = torch.arccos(cosine_angle)

    return angle


def cal_dihedral(a, b, c, d):
    ab = a - b
    cb = c - b
    dc = d - c

    cb = cb / torch.norm(cb, p=2, dim=1).reshape(-1, 1)
    v = ab - torch.matmul(ab.unsqueeze(-1).transpose(-2, -1), cb.unsqueeze(-1)).squeeze(-1) * cb
    w = dc - torch.matmul(dc.unsqueeze(-1).transpose(-2, -1), cb.unsqueeze(-1)).squeeze(-1) * cb
    x = torch.matmul(v.unsqueeze(-1).transpose(-2, -1), w.unsqueeze(-1)).squeeze(-1)
    y = torch.matmul(torch.cross(cb, v).unsqueeze(-1).transpose(-2, -1), w.unsqueeze(-1)).squeeze(-1)

    return torch.atan2(y, x)


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = {}
    for full_key, value in current_dict.items():
        name = full_key.replace("layer_dict.", "")
        name = name.replace("block_dict.", "")
        name = name.replace("module-", "")

        parts = name.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            # 当第二段是数字时，把前两段合并为顶层 key，如 "ffi_model.0"
            top_level = parts[0] + "." + parts[1]
            sub_level = ".".join(parts[2:])  # 可能为空字符串
        else:
            # 否则保留第一段作为顶层，如 "predict" 或 "atom_weighting"
            top_level = parts[0]
            sub_level = ".".join(parts[1:])

        # 插入到 output_dict 中，若 sub_level 为空则直接保存 value，否则以 dict 形式保存/扩展
        if sub_level == "" or sub_level is None:
            # 直接将该顶层设置为该值（覆盖式）
            output_dict[top_level] = value
        else:
            # 确保 output_dict[top_level] 是 dict，然后写入子键
            if top_level in output_dict and not isinstance(output_dict[top_level], dict):
                output_dict[top_level] = {"": output_dict[top_level]}
            if top_level not in output_dict:
                output_dict[top_level] = {}
            output_dict[top_level][sub_level] = value

    return output_dict

class MetaLinearLayer(nn.Module):
    def __init__(self, input_shape, num_filters, use_bias):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()
        b, c = input_shape

        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(num_filters, c))
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, weights=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        if weights is not None:
            params = extract_top_level_dict(current_dict=weights)
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
        elif self.use_bias:
            weight, bias = self.weights, self.bias
        else:
            weight = self.weights
            bias = None
        return F.linear(input=x, weight=weight, bias=bias)

class MLP(nn.Module):
    def __init__(self, dim_per_layer: List, dropout=0.0, activation=nn.ELU()):
        super(MLP, self).__init__()
        self.layers = []
        for i in range(len(dim_per_layer) - 2):
            self.layers.append(MetaLinearLayer((None, dim_per_layer[i]), dim_per_layer[i+1], use_bias=True))
            self.layers.append(activation)
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(MetaLinearLayer((None, dim_per_layer[-2]), dim_per_layer[-1], use_bias=True))
        self.model = nn.Sequential(*self.layers)
        self.model.apply(init_weight)

    def forward(self, x: torch.Tensor, weights: dict = None):
        wd = extract_top_level_dict(weights or {})
        for idx, layer in enumerate(self.model):
            if isinstance(layer, MetaLinearLayer):
                subw = wd.get(f"predict.{idx}", None)
                x = layer(x, weights=subw)
            else:
                x = layer(x)
        return x


class PositionEncoder(nn.Module):
    def __init__(self, d_model, seq_len=4, device='cuda:0'):
        super().__init__()
        # position_enc.shape = [seq_len, d_model]
        position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)] for pos in range(seq_len)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
        pos = torch.tensor(position_enc, dtype=torch.float32).unsqueeze(0)
        self.register_buffer('position_enc', pos)

    def forward(self, x):
        pos_enc = self.position_enc
        if pos_enc.device != x.device:
            pos_enc = pos_enc.to(x.device)
        x = x * pos_enc.detach()
        return x