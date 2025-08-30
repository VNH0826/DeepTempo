from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils.layers import create_spectral_features, MLP


class MRSATSPMConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int = 2,
                 self_loop: bool = True, temporal_kernel_size: int = 3,
                 dropout: float = 0.2, use_temporal_conv: bool = True):
        super().__init__()
        self.num_relations = num_relations
        self.self_loop = self_loop
        self.use_temporal_conv = use_temporal_conv

        self.relation_layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)
        ])

        if self_loop:
            self.self_loop_layer = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.self_loop_layer = None

        if self.use_temporal_conv:
            padding = ((temporal_kernel_size - 1) // 2)
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(out_dim, out_dim, kernel_size=temporal_kernel_size, padding=padding),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.temporal_conv = None

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.relation_layers:
            nn.init.xavier_uniform_(layer.weight)
        if self.self_loop:
            nn.init.xavier_uniform_(self.self_loop_layer.weight)

    def forward(self, x: Tensor, edge_indices: list) -> Tensor:
        agg = 0

        for i in range(self.num_relations):
            edge_index = edge_indices[i]
            layer = self.relation_layers[i]

            if edge_index.size(1) == 0:
                continue

            source, target = edge_index[0], edge_index[1]
            messages = layer(x[source])

            agg_per_rel = torch.zeros(x.size(0), messages.size(1),
                                      dtype=messages.dtype, device=x.device)
            agg_per_rel.scatter_add_(0, target.unsqueeze(-1).expand(-1, messages.size(1)), messages)
            agg += agg_per_rel

        if self.self_loop:
            agg += self.self_loop_layer(x)

        if self.temporal_conv is not None and x.size(0) > 3:
            agg_t = agg.transpose(0, 1).unsqueeze(0)
            agg_t = self.temporal_conv(agg_t)
            agg = agg_t.squeeze(0).transpose(0, 1)

        return agg


class ASFR(nn.Module):
    def __init__(self, channels: int, gate_threshold: float = 0.5):
        super().__init__()
        self.gate_threshold = gate_threshold
        self.norm = nn.LayerNorm(channels)
        self.gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.norm(x)
        weights = self.gate(x_norm)

        w1 = torch.where(weights > self.gate_threshold,
                         torch.ones_like(weights), weights)
        w2 = torch.where(weights > self.gate_threshold,
                         torch.zeros_like(weights), weights)

        x1 = w1 * x
        x2 = w2 * x

        x1_1, x1_2 = torch.split(x1, x1.size(1) // 2, dim=1)
        x2_1, x2_2 = torch.split(x2, x2.size(1) // 2, dim=1)

        return torch.cat([x1_1 + x2_2, x1_2 + x2_1], dim=1)


class DeepTempo(nn.Module):
    def __init__(self, args, node_num: int, device: torch.device,
                 in_dim: int = 64, out_dim: int = 64, layer_num: int = 9,
                 temporal_kernel_size: int = 3, dropout: float = 0.2,
                 lamb: float = 5, **kwargs):
        super().__init__(**kwargs)
        self.args = args
        self.node_num = node_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_num = layer_num
        self.lamb = lamb
        self.device = device

        hidden_dim = out_dim // 2

        self.pos_edge_index = None
        self.neg_edge_index = None
        self.x = None

        use_temporal_conv = (layer_num < 7)

        self.conv1 = MRSATSPMConv(
            in_dim, hidden_dim, num_relations=2, dropout=dropout,
            temporal_kernel_size=temporal_kernel_size,
            use_temporal_conv=use_temporal_conv
        )

        self.asfr = ASFR(hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(layer_num - 1):
            self.convs.append(MRSATSPMConv(
                hidden_dim, hidden_dim, num_relations=2, dropout=dropout,
                temporal_kernel_size=temporal_kernel_size,
                use_temporal_conv=use_temporal_conv
            ))

        self.projection = nn.Linear(hidden_dim, out_dim)

        self.readout_prob = MLP(
            out_dim, out_dim, 1, num_layer=3, p_drop=dropout,
            norm_layer='layernorm', act_layer='relu'
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.xavier_uniform_(self.projection.weight)

    def get_x_edge_index(self, init_emb: Tensor, edge_index_s: Tensor):
        pos_mask = edge_index_s[:, 2] > 0
        neg_mask = edge_index_s[:, 2] < 0

        if torch.any(pos_mask):
            self.pos_edge_index = edge_index_s[pos_mask][:, :2].t()
        else:
            self.pos_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        if torch.any(neg_mask):
            self.neg_edge_index = edge_index_s[neg_mask][:, :2].t()
        else:
            self.neg_edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        if init_emb is None:
            init_emb = create_spectral_features(
                pos_edge_index=self.pos_edge_index,
                neg_edge_index=self.neg_edge_index,
                node_num=self.node_num,
                dim=self.in_dim
            ).to(self.device)

        self.x = init_emb

    def forward(self, init_emb: Tensor, edge_index_s: Tensor) -> Tuple[Tensor, Tensor]:
        self.get_x_edge_index(init_emb, edge_index_s)

        z = F.elu(self.conv1(self.x, [self.pos_edge_index, self.neg_edge_index]))
        z = self.asfr(z)

        for i, conv in enumerate(self.convs):
            z_new = F.elu(conv(z, [self.pos_edge_index, self.neg_edge_index]))
            if i > 0:
                z = z_new + 0.1 * z
            else:
                z = z_new

        z = F.elu(self.projection(z))
        prob = torch.sigmoid(self.readout_prob(z))

        return z, prob