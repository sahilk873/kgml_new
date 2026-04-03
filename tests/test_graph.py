from __future__ import annotations

from pathlib import Path
import sys
import types

import networkx as nx
import pandas as pd
import torch

from kgml_new.data.datasets import (
    compute_relation_diversity_buckets,
    prepare_link_prediction_dataset,
)
from kgml_new.data.graph import networkx_to_data
from kgml_new.data.loaders import GraphCSVSpec, load_graph_csv
from kgml_new.embeddings import semantic as semantic_module
from kgml_new.embeddings.semantic import describe_relation, relation_embeddings_from_relation_types


def test_networkx_to_data():
    g = nx.Graph()
    g.add_edge("a", "b", relationship="T")
    g.add_edge("b", "c", relationship="N")
    g.add_edge("c", "a", relationship="T")
    data, relation_lookup = networkx_to_data(g, in_dim=16, seed=0)
    assert data.num_nodes == 3
    assert data.edge_index.shape[1] == 6
    assert data.x.shape == (3, 16)
    assert "T" in relation_lookup
    assert "N" in relation_lookup


def test_load_graph_csv_generic(tmp_path: Path):
    csv_path = tmp_path / "mini_kg.csv"
    df = pd.DataFrame(
        [
            {"src": "drug_a", "dst": "disease_x", "rel": "treats", "src_type": "drug", "dst_type": "disease"},
            {"src": "gene_1", "dst": "drug_a", "rel": "targets", "src_type": "gene", "dst_type": "drug"},
        ]
    )
    df.to_csv(csv_path, index=False)

    spec = GraphCSVSpec(
        source_col="src",
        target_col="dst",
        relation_col="rel",
        source_type_col="src_type",
        target_type_col="dst_type",
    )
    graph = load_graph_csv(csv_path, spec=spec)

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert graph.nodes["drug_a"]["node_type"] == "drug"
    assert graph.nodes["disease_x"]["node_type"] == "disease"
    assert graph["drug_a"]["disease_x"]["relationship"] == "treats"


def test_load_graph_csv_headerless_tsv_drkg_style(tmp_path: Path):
    tsv_path = tmp_path / "mini_drkg.tsv"
    tsv_path.write_text(
        "\n".join(
            [
                "Gene::2157\tbioarx::HumGenHumGen:Gene:Gene\tGene::5264",
                "Compound::DB001\tDRUGBANK::treats\tDisease::DOID:1234",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    spec = GraphCSVSpec(
        source_col="x_name",
        target_col="y_name",
        relation_col="relation",
    )
    graph = load_graph_csv(tsv_path, spec=spec)

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 2
    assert graph.nodes["Gene::2157"]["node_type"] == "gene"
    assert graph.nodes["Compound::DB001"]["node_type"] == "compound"
    assert graph.nodes["Disease::DOID:1234"]["node_type"] == "disease"
    assert graph["Gene::2157"]["Gene::5264"]["relationship"] == "bioarx::HumGenHumGen:Gene:Gene"
    assert graph["Compound::DB001"]["Disease::DOID:1234"]["relationship"] == "DRUGBANK::treats"


def test_describe_relation_prefers_glossary_then_falls_back(tmp_path: Path):
    glossary_path = tmp_path / "relation_glossary.tsv"
    glossary_path.write_text(
        "\t".join(
            [
                "Relation-name",
                "Data-source",
                "Connected entity-types",
                "Interaction-type",
                "Description",
                "Reference for the description",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "bioarx::HumGenHumGen:Gene:Gene",
                "BioARX",
                "Gene:Gene",
                "interaction",
                "Protein-protein interaction",
                "ref",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    glossary_description = describe_relation(
        "bioarx::HumGenHumGen:Gene:Gene",
        glossary_path=glossary_path,
    )
    fallback_description = describe_relation(
        "unknown::rel",
        glossary_path=glossary_path,
    )
    primekg_style_description = describe_relation(
        "drug_protein",
        glossary_path=glossary_path,
    )

    assert "Protein-protein interaction" in glossary_description
    assert "Data source: BioARX." in glossary_description
    assert fallback_description == "Relationship type: unknown::rel"
    assert primekg_style_description == "Relationship type: drug_protein"


def test_legacy_semantic_cache_is_regenerated(tmp_path: Path, monkeypatch):
    cache_path = tmp_path / "legacy-cache.pt"
    torch.save({"rel": torch.ones(8)}, cache_path)

    async def _fake_embed_async(client, texts, model="text-embedding-3-small"):
        return [[0.5] * 16 for _ in texts]

    fake_openai = types.ModuleType("openai")

    class _FakeAsyncOpenAI:
        pass

    fake_openai.AsyncOpenAI = _FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setattr(semantic_module, "_embed_async", _fake_embed_async)

    rel_emb = relation_embeddings_from_relation_types(
        ["rel"],
        edge_dim=8,
        cache_path=cache_path,
        use_openai=True,
    )

    assert rel_emb["rel"].numel() == 16
    saved = torch.load(cache_path)
    assert saved["format_version"] == 2
    assert saved["embedding_dim"] == 16


def test_prepare_link_prediction_dataset_node_split_uses_held_out_nodes():
    g = nx.Graph()
    g.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 0),
            (0, 2),
            (2, 4),
            (1, 3),
        ]
    )
    for u, v in g.edges():
        g[u][v]["relationship"] = "T"

    dataset = prepare_link_prediction_dataset(
        g,
        in_dim=16,
        seed=0,
        val_ratio=0.25,
        test_ratio=0.25,
        split_protocol="node",
    )

    assert dataset.data.num_nodes == 6
    assert dataset.positive_edge_index.shape[0] == 2
    assert dataset.train_data.edge_index.shape[1] == dataset.split.train_pos_edge_index.shape[1] * 2
    assert "UNK" in dataset.relation_lookup
    assert hasattr(dataset.split, "train_node_mask")

    split = dataset.split
    src, dst = dataset.positive_edge_index

    train_edge_mask = split.train_node_mask[src] & split.train_node_mask[dst]
    val_edge_mask = (~(split.test_node_mask[src] | split.test_node_mask[dst])) & (
        split.val_node_mask[src] | split.val_node_mask[dst]
    )
    test_edge_mask = split.test_node_mask[src] | split.test_node_mask[dst]

    assert split.train_pos_edge_index.size(1) == int(train_edge_mask.sum())
    assert split.val_pos_edge_index.size(1) == int(val_edge_mask.sum())
    assert split.test_pos_edge_index.size(1) == int(test_edge_mask.sum())
    assert split.val_pos_edge_index.numel() > 0
    assert split.test_pos_edge_index.numel() > 0

    for edge_index in (split.train_pos_edge_index, split.val_pos_edge_index, split.test_pos_edge_index):
        assert edge_index.shape[0] == 2
        assert edge_index.numel() > 0

    train_src, train_dst = split.train_pos_edge_index
    assert split.train_node_mask[train_src].all()
    assert split.train_node_mask[train_dst].all()

    val_src, val_dst = split.val_pos_edge_index
    assert (split.val_node_mask[val_src] | split.val_node_mask[val_dst]).all()
    assert ~(split.test_node_mask[val_src] | split.test_node_mask[val_dst]).any()

    test_src, test_dst = split.test_pos_edge_index
    assert (split.test_node_mask[test_src] | split.test_node_mask[test_dst]).all()


def test_type_matched_negatives_preserve_destination_type():
    g = nx.Graph()
    g.add_edge("drug_a", "disease_a", relationship="treats")
    g.add_edge("drug_b", "disease_b", relationship="treats")
    g.add_edge("drug_c", "disease_c", relationship="treats")
    g.add_edge("drug_d", "disease_d", relationship="treats")
    g.add_edge("gene_a", "drug_a", relationship="targets")
    for node, node_type in {
        "drug_a": "drug",
        "drug_b": "drug",
        "drug_c": "drug",
        "drug_d": "drug",
        "disease_a": "disease",
        "disease_b": "disease",
        "disease_c": "disease",
        "disease_d": "disease",
        "gene_a": "gene",
        "gene_b": "gene",
        "gene_c": "gene",
    }.items():
        g.add_node(node, node_type=node_type)

    dataset = prepare_link_prediction_dataset(
        g,
        in_dim=16,
        seed=0,
        val_ratio=0.2,
        test_ratio=0.2,
        split_protocol="node",
        negative_sampling_mode="type_matched",
        negatives_per_pos=3,
    )

    split = dataset.split
    assert split.negatives_per_pos == 3
    assert split.test_neg_edge_index.size(1) == split.test_pos_edge_index.size(1) * 3
    positive_keys = {
        tuple(sorted((int(src), int(dst))))
        for src, dst in dataset.positive_edge_index.t().tolist()
    }
    node_types = dataset.node_types
    for pos_idx, (src, dst) in enumerate(split.test_pos_edge_index.t().tolist()):
        pos_dst_type = node_types[int(dst)]
        neg_block = split.test_neg_edge_index[:, pos_idx * 3 : (pos_idx + 1) * 3]
        assert torch.equal(neg_block[0], torch.full((3,), int(src), dtype=torch.long))
        for neg_dst in neg_block[1].tolist():
            assert node_types[int(neg_dst)] == pos_dst_type
            assert tuple(sorted((int(src), int(neg_dst)))) not in positive_keys


def test_relation_diversity_buckets_cover_low_medium_high():
    g = nx.Graph()
    g.add_edge("n0", "a", relationship="r1")
    g.add_edge("n1", "b", relationship="r1")
    g.add_edge("n1", "c", relationship="r2")
    g.add_edge("n1", "c2", relationship="r3")
    g.add_edge("n2", "d", relationship="r1")
    g.add_edge("n2", "e", relationship="r2")
    g.add_edge("n2", "f", relationship="r3")
    g.add_edge("n2", "g", relationship="r4")
    g.add_edge("n2", "h", relationship="r5")
    g.add_edge("n2", "i", relationship="r6")

    node_list = list(g.nodes())
    counts, buckets = compute_relation_diversity_buckets(g, node_list)
    idx = {node: node_list.index(node) for node in node_list}

    assert counts[idx["n0"]] == 1
    assert buckets[idx["n0"]] == "low"
    assert counts[idx["n1"]] == 3
    assert buckets[idx["n1"]] == "medium"
    assert counts[idx["n2"]] == 6
    assert buckets[idx["n2"]] == "high"
