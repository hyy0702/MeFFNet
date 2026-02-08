import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool, global_add_pool
from .model_utils import cal_distance, cal_dihedral, cal_angle, init_weight, MLP, PositionEncoder

from typing import List, Tuple
import numpy as np


def seed_all():
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)


# ==========================================
# 1. RBF Layer
# ==========================================
class RBFLayer(nn.Module):
    def __init__(self, low=0.0, high=10.0, gap=0.25):
        super(RBFLayer, self).__init__()
        self.centers = nn.Parameter(torch.arange(low, high, gap).float(), requires_grad=False)
        self.num_kernels = self.centers.size(0)
        self.beta = nn.Parameter(torch.tensor(10.0 / gap).float(), requires_grad=False)

    def forward(self, radial):
        return torch.exp(-self.beta * torch.square(radial - self.centers))


# ==========================================
# 2. FFiNetModel
# ==========================================
class FFiNetModel(nn.Module):
    def __init__(self, args, input_shape=None):
        super(FFiNetModel, self).__init__()

        self.feature_per_layer = args.feature_per_layer
        self.num_heads = args.num_heads
        self.pred_hidden_dim = args.pred_hidden_dim
        self.pred_dropout = args.pred_dropout
        self.pred_layers = args.pred_layers
        self.activation = args.activation
        self.residual = args.residual
        self.num_tasks = args.num_tasks
        self.bias = args.bias
        self.dropout = args.dropout

        # ablation
        self.no_1hop = getattr(args, 'no_1hop', False)
        self.no_2hop = getattr(args, 'no_2hop', False)
        self.no_3hop = getattr(args, 'no_3hop', False)
        self.no_axial = getattr(args, 'no_axial', False)

        # update phase
        layers = []
        for i in range(len(self.feature_per_layer) - 1):
            layer = FFiLayer(
                num_node_features=self.feature_per_layer[i] * (1 if i == 0 else self.num_heads),
                output_dim=self.feature_per_layer[i + 1],
                num_heads=self.num_heads,
                concat=True if i < len(self.feature_per_layer) - 2 else False,
                activation=self.activation,
                residual=self.residual,
                bias=self.bias,
                dropout=self.dropout,
                no_1hop=self.no_1hop,
                no_2hop=self.no_2hop,
                no_3hop=self.no_3hop,
                no_axial=self.no_axial
            )
            layers.append(layer)
        self.ffi_model = nn.Sequential(*layers)

        # readout phase
        self.atom_weighting = nn.Sequential(
            nn.Linear(self.feature_per_layer[-1], 1),
            nn.Sigmoid()
        )
        self.atom_weighting.apply(init_weight)

        # prediction phase
        self.predict = MLP(([self.feature_per_layer[-1] * 2] + [self.pred_hidden_dim] * self.pred_layers +
                            [self.num_tasks]), dropout=self.pred_dropout)
        self.predict.apply(init_weight)
        self.attention_group = None

    def forward(self, data: Data, **kwargs):
        output, _, _, _, _, _ = self.ffi_model(
            (data.x, data.edge_index, data.triple_index, data.quadra_index, data.pos, data.edge_attr))
        weighted = self.atom_weighting(output)
        output1 = global_max_pool(output, data.batch)
        output2 = global_add_pool(weighted * output, data.batch)
        output = torch.cat([output1, output2], dim=1)
        return self.predict(output)


# ==========================================
# 3. FFiLayer
# ==========================================
class FFiLayer(nn.Module):
    def __init__(self, num_node_features: int, output_dim: int, num_heads: int,
                 activation=nn.PReLU(), concat: bool = True, residual: bool = True,
                 bias: bool = True, dropout: float = 0.1, share_weights: bool = False,
                 no_1hop: bool = False, no_2hop: bool = False, no_3hop: bool = False, no_axial: bool = False):
        super(FFiLayer, self).__init__()

        seed_all()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.residual = residual
        self.activation = activation
        self.concat = concat
        self.dropout = dropout
        self.share_weights = share_weights
        self.no_1hop = no_1hop
        self.no_2hop = no_2hop
        self.no_3hop = no_3hop
        self.no_axial = no_axial

        # Embedding
        self.linear_src = nn.Linear(num_node_features, output_dim * num_heads, bias=False)
        if self.share_weights:
            self.linear_dst = self.linear_src
            self.linear_mid1 = self.linear_src
            self.linear_mid2 = self.linear_src
        else:
            self.linear_dst = nn.Linear(num_node_features, output_dim * num_heads, bias=False)
            self.linear_mid1 = nn.Linear(num_node_features, output_dim * num_heads, bias=False)
            self.linear_mid2 = nn.Linear(num_node_features, output_dim * num_heads, bias=False)

        # --- RBF ---
        self.rbf_fn = RBFLayer(low=0.0, high=10.0, gap=0.25)
        self.num_rbf = self.rbf_fn.num_kernels

        # Distance embedding
        self.linear_pos_bonded = nn.Linear(2, output_dim * num_heads)
        self.linear_pos_unbonded1 = nn.Linear(self.num_rbf, output_dim * num_heads)
        self.linear_pos_unbonded2 = nn.Linear(self.num_rbf, output_dim * num_heads)
        self.linear_pos_unbonded = nn.Linear(self.num_rbf, output_dim * num_heads)

        # Angle/Dihedral embedding
        self.linear_angle = MLP([2, output_dim * num_heads])
        self.linear_dihedral = MLP([6, output_dim * num_heads])

        # Axial attention
        self.linear_one_hop = nn.Linear(output_dim * num_heads, output_dim * num_heads)
        self.linear_two_hop = nn.Linear(output_dim * num_heads, output_dim * num_heads)
        self.linear_three_hop = nn.Linear(output_dim * num_heads, output_dim * num_heads)

        # Learnable attention parameters
        self.double_attn = nn.Parameter(torch.Tensor(1, num_heads, output_dim))
        self.triple_attn = nn.Parameter(torch.Tensor(1, num_heads, output_dim))
        self.quadra_attn = nn.Parameter(torch.Tensor(1, num_heads, output_dim))

        self.layer_norm = nn.LayerNorm(output_dim * num_heads) if concat else nn.LayerNorm(output_dim)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(output_dim * num_heads))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        if residual:
            if num_node_features == num_heads * output_dim:
                self.residual_linear = nn.Identity()
            else:
                self.residual_linear = nn.Linear(num_node_features, num_heads * output_dim, bias=False)
        else:
            self.register_parameter('residual_linear', None)

        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.dropout_layer = nn.Dropout(dropout)  # Rename to differentiate from param

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_src.weight)
        nn.init.xavier_uniform_(self.linear_dst.weight)
        nn.init.xavier_uniform_(self.linear_mid1.weight)
        nn.init.xavier_uniform_(self.linear_mid2.weight)
        self.linear_one_hop.apply(init_weight)
        self.linear_two_hop.apply(init_weight)
        self.linear_three_hop.apply(init_weight)
        self.linear_pos_bonded.apply(init_weight)
        self.linear_pos_unbonded.apply(init_weight)
        self.linear_pos_unbonded1.apply(init_weight)
        self.linear_pos_unbonded2.apply(init_weight)
        self.linear_angle.apply(init_weight)
        self.linear_dihedral.apply(init_weight)
        nn.init.xavier_uniform_(self.double_attn)
        nn.init.xavier_uniform_(self.triple_attn)
        nn.init.xavier_uniform_(self.quadra_attn)
        if self.residual and isinstance(self.residual_linear, nn.Linear):
            nn.init.xavier_uniform_(self.residual_linear.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        x, edge_index, triple_index, quadra_index, pos, edge_attr = data
        dihedral_src_index, dihedral_mid2_index, dihedral_mid1_index, dihedral_dst_index = quadra_index
        angle_src_index, angle_mid_index, angle_dst_index = triple_index
        edge_src_index, edge_dst_index = edge_index

        src_projected = self.linear_src(self.dropout_layer(x))
        dst_projected = self.linear_dst(self.dropout_layer(x))
        mid1_projected = self.linear_mid1(self.dropout_layer(x))
        mid2_projected = self.linear_mid2(self.dropout_layer(x))

        self.position_encoder = PositionEncoder(d_model=self.num_heads * self.output_dim, device=x.device)
        x_squence = torch.stack([src_projected, mid2_projected, mid1_projected, dst_projected], dim=1)
        x_squence = self.position_encoder(x_squence)

        src_projected = x_squence[:, 0, :].view(-1, self.num_heads, self.output_dim)
        mid2_projected = x_squence[:, 1, :].view(-1, self.num_heads, self.output_dim)
        mid1_projected = x_squence[:, 2, :].view(-1, self.num_heads, self.output_dim)
        dst_projected = x_squence[:, 3, :].view(-1, self.num_heads, self.output_dim)

        num_nodes = x.shape[0]

        # === 1-hop ===
        if not self.no_1hop:
            src_pos, dst_pos = pos.index_select(0, edge_src_index), pos.index_select(0, edge_dst_index)
            distance_per_edge = cal_distance(src_pos, dst_pos).unsqueeze(-1)

            # RBF
            distance_matrix_bonded = torch.cat([distance_per_edge, distance_per_edge ** 2], dim=1)
            dist_clamped = distance_per_edge.clamp(min=0.05, max=10.0)
            distance_matrix_unbonded = self.rbf_fn(dist_clamped)  # RBF feature

            if edge_attr is None:
                edge_attr = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=x.device)

            distance_matrix = self.linear_pos_bonded(distance_matrix_bonded * ~edge_attr.unsqueeze(-1)) + \
                              self.linear_pos_unbonded(distance_matrix_unbonded * edge_attr.unsqueeze(-1))
            distance_matrix = distance_matrix.view(-1, self.num_heads, self.output_dim)

            edge_attn = self.leakyReLU((mid1_projected.index_select(0, edge_src_index)
                                        + dst_projected.index_select(0, edge_dst_index)) * distance_matrix)
            edge_attn = (self.double_attn * edge_attn).sum(-1)
            exp_edge_attn = (edge_attn - edge_attn.max()).exp()

            edge_node_score_sum = torch.zeros([num_nodes, self.num_heads], dtype=exp_edge_attn.dtype,
                                              device=exp_edge_attn.device)
            edge_node_score_sum.scatter_add_(0, edge_dst_index.unsqueeze(-1).expand_as(exp_edge_attn), exp_edge_attn)
            exp_edge_attn = exp_edge_attn / (edge_node_score_sum.index_select(0, edge_dst_index) + 1e-16)
            exp_edge_attn = self.dropout_layer(exp_edge_attn).unsqueeze(-1)

            edge_x_projected = mid1_projected.index_select(0, edge_src_index) * exp_edge_attn
            edge_output = torch.zeros([num_nodes, self.num_heads, self.output_dim], dtype=exp_edge_attn.dtype,
                                      device=exp_edge_attn.device)
            edge_output.scatter_add_(0, edge_dst_index.unsqueeze(-1).unsqueeze(-1).expand_as(edge_x_projected),
                                     edge_x_projected)
        else:
            edge_output = torch.zeros([num_nodes, self.num_heads, self.output_dim],
                                      dtype=x.dtype, device=x.device)

        # === 2-hop ===
        if not self.no_2hop:
            angle_src_pos, angle_dst_pos = pos.index_select(0, angle_src_index), pos.index_select(0, angle_dst_index)
            distance_per_angle = cal_distance(angle_src_pos, angle_dst_pos).unsqueeze(-1)

            # RBF
            dist_angle_clamped = distance_per_angle.clamp(min=0.05, max=10.0)
            distance_matrix_angle = self.rbf_fn(dist_angle_clamped)  # RBF
            distance_matrix_angle = self.linear_pos_unbonded1(distance_matrix_angle).view(-1, self.num_heads,
                                                                                          self.output_dim)

            angle_src_pos = pos.index_select(0, angle_src_index)
            angle_mid_pos = pos.index_select(0, angle_mid_index)
            angle_dst_pos = pos.index_select(0, angle_dst_index)
            angle_per_triedge = cal_angle(angle_src_pos, angle_mid_pos, angle_dst_pos)
            angle_matrix = torch.cat([angle_per_triedge, angle_per_triedge ** 2], dim=1)
            angle_matrix = self.linear_angle(angle_matrix).view(-1, self.num_heads, self.output_dim)

            angle_attn = self.leakyReLU((mid2_projected.index_select(0, angle_src_index)
                                         + dst_projected.index_select(0, angle_dst_index)
                                         + mid1_projected.index_select(0, angle_mid_index))
                                        * (angle_matrix + distance_matrix_angle))
            angle_attn = ((self.triple_attn * angle_attn).sum(-1))
            exp_angle_attn = (angle_attn - angle_attn.max()).exp()

            angle_node_score_sum = torch.zeros([num_nodes, self.num_heads], dtype=exp_angle_attn.dtype,
                                               device=exp_angle_attn.device)
            angle_node_score_sum.scatter_add_(0, angle_dst_index.unsqueeze(-1).expand_as(exp_angle_attn), exp_angle_attn)
            exp_angle_attn = exp_angle_attn / (angle_node_score_sum.index_select(0, angle_dst_index) + 1e-16)
            exp_angle_attn = self.dropout_layer(exp_angle_attn).unsqueeze(-1)

            angle_x_projected = mid2_projected.index_select(0, angle_src_index) * exp_angle_attn
            angle_output = torch.zeros([num_nodes, self.num_heads, self.output_dim], dtype=exp_angle_attn.dtype,
                                       device=exp_angle_attn.device)
            angle_output.scatter_add_(0, angle_dst_index.unsqueeze(-1).unsqueeze(-1).expand_as(angle_x_projected),
                                      angle_x_projected)
        else:
            angle_output = torch.zeros([num_nodes, self.num_heads, self.output_dim],
                                       dtype=x.dtype, device=x.device)

        # === 3-hop ===
        if not self.no_3hop:
            dihedral_src_pos = pos.index_select(0, dihedral_src_index)
            dihedral_dst_pos = pos.index_select(0, dihedral_dst_index)
            distance_per_dihedral = cal_distance(dihedral_src_pos, dihedral_dst_pos).unsqueeze(-1)

            # RBF
            dist_dihedral_clamped = distance_per_dihedral.clamp(min=0.05, max=10.0)
            distance_matrix_dihedral = self.rbf_fn(dist_dihedral_clamped)  # RBF
            distance_matrix_dihedral = self.linear_pos_unbonded2(distance_matrix_dihedral).view(-1, self.num_heads,
                                                                                                self.output_dim)

            dihedral_mid1_pos = pos.index_select(0, dihedral_mid1_index)
            dihedral_mid2_pos = pos.index_select(0, dihedral_mid2_index)
            dihedral_per_quaedge = cal_dihedral(dihedral_src_pos, dihedral_mid2_pos, dihedral_mid1_pos, dihedral_dst_pos)
            dihedral_matrix = torch.cat(
                [torch.cos(dihedral_per_quaedge), torch.cos(dihedral_per_quaedge * 2), torch.cos(dihedral_per_quaedge * 3),
                 torch.sin(dihedral_per_quaedge), torch.sin(dihedral_per_quaedge * 2), torch.sin(dihedral_per_quaedge * 3)],
                dim=1)
            dihedral_matrix = self.linear_dihedral(dihedral_matrix).view(-1, self.num_heads, self.output_dim)

            dihedral_attn = self.leakyReLU((src_projected.index_select(0, dihedral_src_index)
                                            + dst_projected.index_select(0, dihedral_dst_index)
                                            + mid1_projected.index_select(0, dihedral_mid1_index)
                                            + mid2_projected.index_select(0, dihedral_mid2_index))
                                           * (dihedral_matrix + distance_matrix_dihedral))
            dihedral_attn = ((self.quadra_attn * dihedral_attn).sum(-1))
            exp_dihedral_attn = (dihedral_attn - dihedral_attn.max()).exp()

            dihedral_node_score_sum = torch.zeros([num_nodes, self.num_heads], dtype=exp_dihedral_attn.dtype,
                                                  device=exp_dihedral_attn.device)
            dihedral_node_score_sum.scatter_add_(0, dihedral_dst_index.unsqueeze(-1).expand_as(exp_dihedral_attn),
                                                 exp_dihedral_attn)
            exp_dihedral_attn = exp_dihedral_attn / (dihedral_node_score_sum.index_select(0, dihedral_dst_index) + 1e-16)
            exp_dihedral_attn = self.dropout_layer(exp_dihedral_attn).unsqueeze(-1)

            dihedral_x_projected = src_projected.index_select(0, dihedral_src_index) * exp_dihedral_attn
            dihedral_output = torch.zeros([num_nodes, self.num_heads, self.output_dim], dtype=exp_dihedral_attn.dtype,
                                          device=exp_dihedral_attn.device)
            dihedral_output.scatter_add_(0, dihedral_dst_index.unsqueeze(-1).unsqueeze(-1).expand_as(dihedral_x_projected),
                                         dihedral_x_projected)
        else:
            dihedral_output = torch.zeros([num_nodes, self.num_heads, self.output_dim],
                                          dtype=x.dtype, device=x.device)

        # === Axial ===
        if self.no_axial:
            output = edge_output + angle_output + dihedral_output
        else:
            one_hop = self.linear_one_hop(edge_output.view(-1, self.num_heads * self.output_dim)).view(-1, self.num_heads,
                                                                                                       self.output_dim)
            two_hop = self.linear_two_hop(angle_output.view(-1, self.num_heads * self.output_dim)).view(-1, self.num_heads,
                                                                                                        self.output_dim)
            three_hop = self.linear_three_hop(dihedral_output.view(-1, self.num_heads * self.output_dim)).view(-1,
                                                                                                               self.num_heads,
                                                                                                               self.output_dim)

            zero_hop = dst_projected

            one_hop_attn = torch.diagonal(torch.matmul(zero_hop, one_hop.transpose(-2, -1)), dim1=-1, dim2=-2) / np.sqrt(
                self.output_dim)
            two_hop_attn = torch.diagonal(torch.matmul(zero_hop, two_hop.transpose(-2, -1)), dim1=-1, dim2=-2) / np.sqrt(
                self.output_dim)
            three_hop_attn = torch.diagonal(torch.matmul(zero_hop, three_hop.transpose(-2, -1)), dim1=-1,
                                            dim2=-2) / np.sqrt(self.output_dim)

            squence_attn = torch.stack([one_hop_attn, two_hop_attn, three_hop_attn], dim=0)
            squence_attn = self.dropout_layer(torch.softmax(squence_attn, dim=0).unsqueeze(-1))

            output = squence_attn[0, :, :] * edge_output + squence_attn[1, :, :] * angle_output + squence_attn[2, :,
                                                                                                  :] * dihedral_output

        # residual, concat, bias, activation
        if self.residual:
            output += self.residual_linear(x).view(-1, self.num_heads, self.output_dim)
        if self.concat:
            output = output.view(-1, self.num_heads * self.output_dim)
        else:
            output = output.mean(dim=1)
        if self.bias is not None:
            output += self.bias
        output = self.layer_norm(output)
        if self.activation is not None:
            output = self.activation(output)

        return output, edge_index, triple_index, quadra_index, pos, edge_attr