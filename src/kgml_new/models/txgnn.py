from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


class TxGNN(nn.Module):
    """
    PyG implementation inspired by TxGNN:
    - relation-aware heterogeneous message passing
    - DistMult-style relation decoder
    - disease prototype augmentation for zero-shot-like transfer
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_prototypes: bool = True,
        prototype_k: int = 5,
        prototype_alpha: float = 0.5,
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata
        self.edge_types = edge_types
        self.dropout = dropout
        self.use_prototypes = use_prototypes
        self.prototype_k = prototype_k
        self.prototype_alpha = prototype_alpha

        self.input_linears = nn.ModuleDict(
            {ntype: nn.Linear(in_channels, hidden_channels) for ntype in node_types}
        )

        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = hidden_channels if layer_idx == 0 else out_channels
            out_dim = out_channels
            conv_dict = {
                etype: SAGEConv((in_dim, in_dim), out_dim, normalize=False)
                for etype in edge_types
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        self.rel_emb = nn.ParameterDict(
            {
                self._etype_key(etype): nn.Parameter(torch.empty(out_channels))
                for etype in edge_types
            }
        )
        for param in self.rel_emb.values():
            nn.init.xavier_uniform_(param.unsqueeze(0))

    @staticmethod
    def _etype_key(etype: tuple[str, str, str]) -> str:
        return f"{etype[0]}__{etype[1]}__{etype[2]}"

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = {k: self.input_linears[k](v.x) for k, v in data.node_items()}
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, data.edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {
                    k: F.dropout(F.leaky_relu(v), p=self.dropout, training=self.training)
                    for k, v in x_dict.items()
                }
        return {k: F.normalize(v, p=2, dim=-1) for k, v in x_dict.items()}

    def _prototype_augment(
        self,
        disease_embeddings: torch.Tensor,
        query_indices: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_prototypes or disease_embeddings.size(0) <= 1:
            return disease_embeddings[query_indices]
        query = disease_embeddings[query_indices]
        sim = query @ disease_embeddings.t()
        k = min(self.prototype_k + 1, disease_embeddings.size(0))
        topk = torch.topk(sim, k=k, dim=1)
        neighbor_idx = topk.indices[:, 1:] if k > 1 else topk.indices
        neighbor_w = topk.values[:, 1:] if k > 1 else topk.values
        neighbor_w = F.softmax(neighbor_w, dim=1)
        proto = (disease_embeddings[neighbor_idx] * neighbor_w.unsqueeze(-1)).sum(dim=1)
        return (1.0 - self.prototype_alpha) * query + self.prototype_alpha * proto

    def score_edges(
        self,
        z_dict: dict[str, torch.Tensor],
        edge_type: tuple[str, str, str],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        src_type, _, dst_type = edge_type
        src_z = z_dict[src_type][edge_index[0]]
        dst_indices = edge_index[1]

        if dst_type == "disease":
            dst_z = self._prototype_augment(z_dict[dst_type], dst_indices)
        else:
            dst_z = z_dict[dst_type][dst_indices]

        rel = self.rel_emb[self._etype_key(edge_type)]
        return torch.sum(src_z * rel * dst_z, dim=-1)
