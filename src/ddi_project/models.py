from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

from .config import DEFAULT_CONFIG


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight[targets]
        return loss.mean()


class SiameseGCN(nn.Module):
    def __init__(self, n_classes: int, hidden: int = 256, dropout: float = 0.30):
        super().__init__()
        self.conv1 = GCNConv(DEFAULT_CONFIG.atom_feature_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def encode(self, graph) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.conv1(graph.x, graph.edge_index))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x, graph.edge_index))))
        x = self.conv3(x, graph.edge_index)
        return global_mean_pool(x, graph.batch)

    def forward(self, graph_a, graph_b) -> torch.Tensor:
        return self.classifier(torch.cat([self.encode(graph_a), self.encode(graph_b)], dim=-1))


class DeepDDI(nn.Module):
    def __init__(self, n_classes: int, fp_dim: int = DEFAULT_CONFIG.fp_dim, hidden: int = 2048, dropout: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, fp_a: torch.Tensor, fp_b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([fp_a, fp_b], dim=-1))


class GATEncoder(nn.Module):
    def __init__(self, atom_dim: int, fp_dim: int, hidden: int, heads: int = 4, dropout: float = 0.20):
        super().__init__()
        self.conv1 = GATConv(atom_dim, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv3 = GATConv(hidden * heads, hidden, heads=1, dropout=dropout, concat=False)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden * heads)
        self.dropout = nn.Dropout(dropout)
        self.fp_proj = nn.Sequential(nn.Linear(fp_dim, hidden), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x, edge_index, batch, fp):
        x = self.dropout(F.elu(self.norm1(self.conv1(x, edge_index))))
        x = self.dropout(F.elu(self.norm2(self.conv2(x, edge_index))))
        x = self.conv3(x, edge_index)
        graph_embedding = global_mean_pool(x, batch)
        return graph_embedding + self.fp_proj(fp)


class CrossDrugCoAttention(nn.Module):
    def __init__(self, hidden: int, attn_dim: int = 128):
        super().__init__()
        self.W_q = nn.Linear(hidden, attn_dim, bias=False)
        self.W_k = nn.Linear(hidden, attn_dim, bias=False)
        self.W_v = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden * 2, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        scale = self.W_q(h_a).size(-1) ** 0.5
        alpha_a = torch.sigmoid((self.W_q(h_a) * self.W_k(h_b)).sum(-1, keepdim=True) / scale)
        alpha_b = torch.sigmoid((self.W_q(h_b) * self.W_k(h_a)).sum(-1, keepdim=True) / scale)
        attended_a = alpha_a * self.W_v(h_b)
        attended_b = alpha_b * self.W_v(h_a)
        return self.norm(self.out(torch.cat([attended_a, attended_b], dim=-1)))


class KGSSGNN(nn.Module):
    def __init__(
        self,
        n_classes: int,
        atom_dim: int = DEFAULT_CONFIG.atom_feature_dim,
        fp_dim: int = DEFAULT_CONFIG.fp_dim,
        kg_dim: int = DEFAULT_CONFIG.kg_dim,
        hidden: int = DEFAULT_CONFIG.hidden_dim,
        heads: int = DEFAULT_CONFIG.gat_heads,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.gat = GATEncoder(atom_dim, fp_dim, hidden, heads, dropout)
        self.co_attn = CrossDrugCoAttention(hidden, attn_dim=128)
        self.kg_proj = nn.Sequential(
            nn.Linear(kg_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 5, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, graph_a, graph_b, fp_a, fp_b, kg_a, kg_b) -> torch.Tensor:
        h_a = self.gat(graph_a.x, graph_a.edge_index, graph_a.batch, fp_a)
        h_b = self.gat(graph_b.x, graph_b.edge_index, graph_b.batch, fp_b)
        co = self.co_attn(h_a, h_b)
        k_a = self.kg_proj(kg_a)
        k_b = self.kg_proj(kg_b)
        return self.classifier(torch.cat([h_a, h_b, co, k_a, k_b], dim=-1))
