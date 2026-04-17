from __future__ import annotations

"""
Heterogeneous Graph Transformer link encoder (Hu et al., WWW 2020).

Uses PyG :class:`torch_geometric.nn.conv.HGTConv` with a DistMult-style
relation decoder. Independent of :class:`kgml_new.models.txgnn.TxGNN`.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv


def _channels_divisible_by_heads(channels: int, heads: int) -> int:
    """HGTConv requires out_channels % heads == 0."""
    if heads <= 0:
        raise ValueError("heads must be positive")
    aligned = (channels // heads) * heads
    if aligned == 0:
        raise ValueError(f"channels ({channels}) too small for heads ({heads})")
    return aligned


class HGTLinkPredictor(nn.Module):
    """
    Stacked HGTConv layers + L2-normalized per-type embeddings and DistMult scores.
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata
        self.edge_types = edge_types
        self.dropout = dropout

        hidden_channels = _channels_divisible_by_heads(hidden_channels, num_heads)
        out_channels = _channels_divisible_by_heads(out_channels, num_heads)

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.input_linears = nn.ModuleDict(
            {ntype: nn.Linear(in_channels, hidden_channels) for ntype in node_types}
        )

        self.convs = nn.ModuleList()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        for layer_idx in range(num_layers):
            if layer_idx < num_layers - 1:
                self.convs.append(
                    HGTConv(
                        hidden_channels,
                        hidden_channels,
                        metadata,
                        heads=num_heads,
                    )
                )
            else:
                self.convs.append(
                    HGTConv(
                        hidden_channels,
                        out_channels,
                        metadata,
                        heads=num_heads,
                    )
                )

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
                next_dict: dict[str, torch.Tensor] = {}
                for k, v in x_dict.items():
                    if v is None:
                        raise RuntimeError(
                            f"HGT layer {i} returned None for node type {k!r} (disconnected type?)"
                        )
                    next_dict[k] = F.dropout(
                        F.leaky_relu(v), p=self.dropout, training=self.training
                    )
                x_dict = next_dict
        out: dict[str, torch.Tensor] = {}
        for k, v in x_dict.items():
            if v is None:
                raise RuntimeError(f"HGT produced None embedding for node type {k!r}")
            out[k] = F.normalize(v, p=2, dim=-1)
        return out

    def score_edges(
        self,
        z_dict: dict[str, torch.Tensor],
        edge_type: tuple[str, str, str],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        src_type, _, dst_type = edge_type
        src_z = z_dict[src_type][edge_index[0]]
        dst_z = z_dict[dst_type][edge_index[1]]
        rel = self.rel_emb[self._etype_key(edge_type)]
        return torch.sum(src_z * rel * dst_z, dim=-1)
