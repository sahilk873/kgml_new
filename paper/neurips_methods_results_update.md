# NeurIPS-Style Methods and Results Draft

This file is a publication-oriented replacement draft for the current `Methods` section and the incomplete `Results` tables in `paper/Lexical Semantic Context Injection Improves Graph Representation Learning Performance at Low Computational Cost.docx`. It is grounded in the current `kgml_new` implementation and the completed result artifacts under `results/slurm`.

## Methods

### Problem Setup

We study link prediction on heterogeneous knowledge graphs with many relation types. Given a graph \(G=(V,E)\) with typed relations \(r \in \mathcal{R}\), the task is to score whether an edge \((u,r,v)\) should exist. We evaluate both a transductive edge-split setting and a more challenging inductive node-split setting. In the edge split, all nodes may appear during training and only positive edges are withheld for validation and test. In the node split, nodes are partitioned into train, validation, and test subsets; training positives connect only training nodes, while validation and test edges involve held-out nodes. Following the repository defaults, negative edges are sampled globally for the edge split and type-matched for the node split.

### Datasets

We report experiments on PrimeKG and FB15k-237. PrimeKG is a biomedical knowledge graph with typed entities and semantically meaningful biomedical relations. FB15k-237 is a standard multi-relational benchmark used for link prediction. Both datasets are loaded through the same `run_gpu_method` pipeline, which constructs homogeneous PyG graphs with relation labels on edges and evaluates all methods under matched splits and metrics.

### Lexical Semantic Edge Context Injection

Our method augments GraphSAGE with frozen semantic representations of relation types. For each relation type \(r\), we build a text description \(t_r\). For DRKG-style sourced predicates this description can include glossary-derived fields such as relation name, source database, connected entity types, interaction type, and free-text description. For PrimeKG-style predicates, the raw relation string is embedded directly. We then map each relation text \(t_r\) to a dense vector \(e_r\) using one of three text embedding backends used in our experiments: OpenAI, Gemini, or E5.

The semantic relation table is computed once per relation vocabulary and cached. During graph learning, the relation table is frozen. If the embedding dimensionality of the chosen text encoder differs from the message-passing edge dimension, the model learns a single linear projection into the edge message space. This design keeps the semantic prior fixed while allowing the graph model to align relation semantics with task-specific representations at low computational cost.

### Edge-Aware GraphSAGE

Our main encoder is an edge-aware variant of GraphSAGE implemented in `src/kgml_new/models/edge_aware_sage.py`. Let \(h_u^{(\ell)}\) be the node representation of node \(u\) at layer \(\ell\), and let \(\tilde e_r\) denote the projected semantic embedding of relation \(r\). In the `concat` variant used for all PrimeKG and FB15k-237 results reported here, each incoming message is computed as

\[
m_{u \rightarrow v}^{(\ell)} = W^{(\ell)} [h_u^{(\ell)} \, \| \, \tilde e_r].
\]

Messages are aggregated with mean pooling over the neighborhood and combined with a learned root transform,

\[
h_v^{(\ell+1)} = \mathrm{AGG}_{u \in \mathcal{N}(v)} m_{u \rightarrow v}^{(\ell)} + W_{\mathrm{root}}^{(\ell)} h_v^{(\ell)}.
\]

This mirrors the self-term behavior of standard GraphSAGE: if a node has no incident edges in the sampled neighborhood, it still receives the root transformation. The final hop is also relation-aware rather than reverting to a relation-blind readout layer. Output node embeddings are L2-normalized before link scoring.

Although the codebase also supports gated, basis-mixture, and FiLM-style relation control, the runs reported in this paper use the `concat` mode in order to isolate the effect of semantic relation initialization.

### Baselines

We compare against four baselines already implemented in the repository.

1. `baseline_gcn`: a homogeneous GCN link predictor.
2. `baseline_sage`: standard GraphSAGE without relation semantics.
3. `node2vec`: a transductive random-walk embedding baseline.
4. `hgt`: a heterogeneous graph transformer baseline (`HGTConv`) evaluated where completed result artifacts are available.

For fairness, the semantic edge-aware model and the GraphSAGE baseline share the same core training pipeline and differ primarily in whether relation semantics are injected into the message function.

### Training Protocol

All reported GraphSAGE and GCN runs use the common `run_gpu_method` pipeline. Unless otherwise stated, the default configuration from `src/kgml_new/config.py` is used: input dimension 64, edge dimension 32, hidden dimension 32, output dimension 64, 2 message-passing layers, 100 epochs, learning rate \(10^{-3}\), batch size 256, and mean neighborhood aggregation with fanouts `[10, 10]`. Evaluation uses 20 negatives per positive edge. Link quality is measured with ROC-AUC, average precision (AP), Hits@1, Hits@3, and Hits@10.

For embedding ablations, we compare OpenAI, Gemini, and E5 as relation encoders while keeping the downstream graph learner fixed. We exclude the separate `edge_aware_sage_node_emb` variant from the main paper tables because it injects pretrained node features in addition to edge semantics and therefore changes the modeling assumption.

## Results

### Main Comparison

Tables 1 and 2 report the main comparison between standard graph baselines and our best edge-aware semantic model for each dataset and split. For the edge-aware model, we report the best AP among the completed OpenAI, Gemini, and E5 runs for that dataset/split and identify the winning embedding backend in parentheses. We omit Node2Vec from the node-split table because it is transductive and the completed node-split artifacts do not provide a meaningful inductive comparison. We also omit FB15k-237 HGT because no completed HGT result JSON for that dataset is present in `results/slurm`.

### Table 1. Edge-split link prediction

| Dataset | Method | ROC-AUC | AP | Hits@1 | Hits@3 | Hits@10 |
|---|---|---:|---:|---:|---:|---:|
| PrimeKG | Baseline GCN | 0.684 | 0.085 | 0.113 | 0.296 | 0.753 |
| PrimeKG | Baseline GraphSAGE | 0.906 | 0.244 | 0.347 | 0.737 | 0.970 |
| PrimeKG | GraphSAGE-Edge (OpenAI) | 0.946 | 0.511 | 0.553 | 0.867 | 0.985 |
| PrimeKG | HGT | 0.962 | 0.365 | - | - | - |
| PrimeKG | Node2Vec | 0.958 | 0.651 | 0.757 | 0.901 | 0.981 |
| FB15k-237 | Baseline GCN | 0.623 | 0.089 | 0.144 | 0.306 | 0.645 |
| FB15k-237 | Baseline GraphSAGE | 0.846 | 0.283 | 0.344 | 0.627 | 0.900 |
| FB15k-237 | GraphSAGE-Edge (Gemini) | 0.896 | 0.393 | 0.456 | 0.746 | 0.949 |
| FB15k-237 | Node2Vec | 0.807 | 0.131 | 0.347 | 0.629 | 0.907 |

### Table 2. Node-split link prediction

| Dataset | Method | ROC-AUC | AP | Hits@1 | Hits@3 | Hits@10 |
|---|---|---:|---:|---:|---:|---:|
| PrimeKG | Baseline GCN | 0.679 | 0.084 | 0.103 | 0.286 | 0.764 |
| PrimeKG | Baseline GraphSAGE | 0.879 | 0.219 | 0.288 | 0.669 | 0.947 |
| PrimeKG | GraphSAGE-Edge (Gemini) | 0.938 | 0.478 | 0.497 | 0.834 | 0.974 |
| PrimeKG | HGT | 0.841 | 0.450 | - | - | - |
| FB15k-237 | Baseline GCN | 0.558 | 0.058 | 0.091 | 0.219 | 0.572 |
| FB15k-237 | Baseline GraphSAGE | 0.804 | 0.199 | 0.266 | 0.542 | 0.862 |
| FB15k-237 | GraphSAGE-Edge (OpenAI) | 0.881 | 0.334 | 0.409 | 0.704 | 0.932 |

### Embedding Ablation for GraphSAGE-Edge

Tables 3 and 4 break out the semantic edge-aware results by relation embedding backend. These tables make clear that the gain over standard GraphSAGE is not tied to a single provider-specific embedding model. Across both datasets and both split protocols, all three semantic encoders materially outperform the relation-agnostic GraphSAGE baseline on AP. The best backend differs by dataset and split: OpenAI gives the best AP on PrimeKG edge and FB15k-237 node, Gemini is strongest on PrimeKG node and FB15k-237 edge, and E5 remains competitive throughout.

### Table 3. Edge-split GraphSAGE-Edge ablation by relation embedding

| Dataset | Relation Embedding | ROC-AUC | AP | Hits@1 | Hits@3 | Hits@10 |
|---|---|---:|---:|---:|---:|---:|
| PrimeKG | OpenAI | 0.946 | 0.511 | 0.553 | 0.867 | 0.985 |
| PrimeKG | Gemini | 0.946 | 0.504 | 0.544 | 0.864 | 0.983 |
| PrimeKG | E5 | 0.947 | 0.509 | 0.548 | 0.866 | 0.985 |
| FB15k-237 | OpenAI | 0.895 | 0.382 | 0.452 | 0.745 | 0.950 |
| FB15k-237 | Gemini | 0.896 | 0.393 | 0.456 | 0.746 | 0.949 |
| FB15k-237 | E5 | 0.891 | 0.367 | 0.443 | 0.735 | 0.945 |

### Table 4. Node-split GraphSAGE-Edge ablation by relation embedding

| Dataset | Relation Embedding | ROC-AUC | AP | Hits@1 | Hits@3 | Hits@10 |
|---|---|---:|---:|---:|---:|---:|
| PrimeKG | OpenAI | 0.938 | 0.473 | 0.487 | 0.829 | 0.975 |
| PrimeKG | Gemini | 0.938 | 0.478 | 0.497 | 0.834 | 0.974 |
| PrimeKG | E5 | 0.937 | 0.469 | 0.491 | 0.831 | 0.973 |
| FB15k-237 | OpenAI | 0.881 | 0.334 | 0.409 | 0.704 | 0.932 |
| FB15k-237 | Gemini | 0.876 | 0.330 | 0.402 | 0.695 | 0.927 |
| FB15k-237 | E5 | 0.872 | 0.323 | 0.398 | 0.690 | 0.919 |

### Result Summary Text

The main empirical finding is that lexical semantic relation injection consistently improves GraphSAGE over a relation-agnostic backbone at very low architectural cost. On PrimeKG, GraphSAGE-Edge improves AP from 0.244 to 0.511 in the edge split and from 0.219 to 0.478 in the node split, more than doubling AP in both cases. On FB15k-237, AP improves from 0.283 to 0.393 in the edge split and from 0.199 to 0.334 in the node split. These gains are accompanied by substantial improvements in ROC-AUC and Hits@K, indicating that the semantic edge signal helps both ranking quality and top-k retrieval.

The embedding ablation suggests that the benefit is robust across multiple text encoders rather than being specific to one proprietary model. OpenAI, Gemini, and E5 all outperform vanilla GraphSAGE on both datasets. The relative ranking of the three encoders changes across datasets and split protocols, which suggests that lexical relation injection is the dominant effect, while the exact text encoder mostly modulates performance within a narrower range.

Compared with HGT, the picture is more nuanced. On PrimeKG edge split, HGT achieves the highest ROC-AUC but substantially lower AP than GraphSAGE-Edge, implying that the transformer baseline ranks negatives reasonably well overall but is less effective than semantic edge injection at concentrating positives near the top of the ranking. On the more difficult PrimeKG node split, GraphSAGE-Edge also exceeds HGT in both ROC-AUC and AP. This supports the claim that lightweight semantic edge conditioning can be competitive with, and in some regimes preferable to, more expensive heterogeneous architectures.

Node2Vec remains strong on the transductive PrimeKG edge split, where it benefits from unrestricted access to node identities and graph structure, but it is not an appropriate baseline for the inductive node split. For this reason, the strongest evidence for our claim comes from the node-disjoint setting, where GraphSAGE-Edge preserves a simple GraphSAGE training recipe while substantially improving generalization to held-out nodes.

## Provenance of Reported Numbers

All numbers above were taken from completed JSON artifacts already present in this repository:

- `results/slurm/primekg_edge_baseline_gcn_3249660_1.json`
- `results/slurm/primekg_edge_baseline_sage_3249660_2.json`
- `results/slurm/primekg_edge_edge_aware_sage_concat_openai_3252764_3.json`
- `results/slurm/primekg_edge_edge_aware_sage_concat_gemini_3252847_4.json`
- `results/slurm/primekg_edge_edge_aware_sage_concat_e5_3252764_5.json`
- `results/slurm/primekg_edge_node2vec_3253102_0.json`
- `results/slurm/primekg_node_baseline_gcn_3249660_7.json`
- `results/slurm/primekg_node_baseline_sage_3249660_8.json`
- `results/slurm/primekg_node_edge_aware_sage_concat_openai_3252764_9.json`
- `results/slurm/primekg_node_edge_aware_sage_concat_gemini_3252847_10.json`
- `results/slurm/primekg_node_edge_aware_sage_concat_e5_3252764_11.json`
- `results/slurm/fb15k237_edge_baseline_gcn_3249660_13.json`
- `results/slurm/fb15k237_edge_baseline_sage_3249660_14.json`
- `results/slurm/fb15k237_edge_edge_aware_sage_concat_openai_3252764_15.json`
- `results/slurm/fb15k237_edge_edge_aware_sage_concat_gemini_3252847_16.json`
- `results/slurm/fb15k237_edge_edge_aware_sage_concat_e5_3252764_17.json`
- `results/slurm/fb15k237_edge_node2vec_3253102_12.json`
- `results/slurm/fb15k237_node_baseline_gcn_3249660_19.json`
- `results/slurm/fb15k237_node_baseline_sage_3249660_20.json`
- `results/slurm/fb15k237_node_edge_aware_sage_concat_openai_3252764_21.json`
- `results/slurm/fb15k237_node_edge_aware_sage_concat_gemini_3252847_22.json`
- `results/slurm/fb15k237_node_edge_aware_sage_concat_e5_3252764_23.json`
- `results/slurm/hgt-primekg-edge-s42-3255705_0.json`
- `results/slurm/hgt-primekg-node-s42-3255705_1.json`

No completed FB15k-237 HGT result JSON is present in `results/slurm`, so those values are intentionally excluded from the publication tables above.
