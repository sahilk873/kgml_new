from __future__ import annotations

import torch
from torch_geometric.data import Data

from kgml_new.data.loaders import GraphCSVSpec
from kgml_new.data.triples import load_triple_kg_dataset
from kgml_new.models.triple_decoders import DistMultTripleDecoder
from kgml_new.training.kg_completion import corrupt_triples, filtered_ranking_metrics


def test_triple_loader_preserves_multiple_relations_between_pair(tmp_path):
    path = tmp_path / "kg.csv"
    path.write_text(
        "src,rel,dst,src_type,dst_type\n"
        "A,r1,B,t1,t2\n"
        "A,r2,B,t1,t2\n"
        "B,r1,C,t2,t2\n"
        "C,r2,A,t2,t1\n"
    )
    ds = load_triple_kg_dataset(
        path,
        spec=GraphCSVSpec(
            source_col="src",
            relation_col="rel",
            target_col="dst",
            source_type_col="src_type",
            target_type_col="dst_type",
        ),
        in_dim=4,
        seed=0,
        val_ratio=0.25,
        test_ratio=0.25,
        split_protocol="edge",
    )
    ab = [
        row.tolist()
        for row in ds.triples
        if ds.node_names[int(row[0])] == "A" and ds.node_names[int(row[2])] == "B"
    ]
    assert len(ab) == 2
    assert len({int(row[1]) for row in ab}) == 2
    assert len(ds.relation_lookup) == 2


def test_corrupt_triples_keeps_relation_fixed():
    triples = torch.tensor([[0, 1, 2], [2, 0, 3]], dtype=torch.long)
    true = {tuple(row.tolist()) for row in triples}
    neg = corrupt_triples(
        triples,
        num_nodes=5,
        true_triples=true,
        negatives_per_pos=4,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    assert neg.shape == (8, 3)
    assert neg[:, 1].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert not any(tuple(row.tolist()) in true for row in neg)


class IdentityEncoder(torch.nn.Module):
    def forward(self, x, edge_index):
        return x


def test_filtered_ranking_filters_other_true_triples():
    data = Data(
        x=torch.eye(3),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=3,
    )
    eval_triples = torch.tensor([[0, 0, 1]], dtype=torch.long)
    all_true = torch.tensor([[0, 0, 1], [0, 0, 2]], dtype=torch.long)
    rel = torch.tensor([[1.0, 1.0, 1.0]])
    decoder = DistMultTripleDecoder(rel, embedding_dim=3)
    metrics = filtered_ranking_metrics(
        IdentityEncoder(),
        decoder,
        data,
        eval_triples,
        all_true,
        node_types=["entity", "entity", "entity"],
        edge_aware=False,
        device=torch.device("cpu"),
    )
    assert metrics["all_entities"]["hits@10"] == 1.0
    assert metrics["type_constrained"]["hits@10"] == 1.0
