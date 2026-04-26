# PrimeKG + FB15k-237 Full Paper Dossier

This file is a consolidated, writer-ready dossier for the PrimeKG and FB15k-237 campaign in `kgml_new`.  
Goal: put all paper-relevant context (methods, splits, run matrix, commands, metrics, artifacts, and interpretation) in one place.

---

## 1) Scope and Ground-Truth Sources

Primary source files used to build this dossier:

- `README.md`
- `EXPERIMENTS.md`
- `results_primekg_fb15k237.md`
- `results.md`
- `paper/neurips_methods_results_update.md`
- `scripts/slurm/primekg_fb15k_build_prepared_caches.slurm`
- `scripts/slurm/primekg_fb15k_full_sweep_array.slurm`
- `scripts/slurm/hgt_primekg_fb15k_split_array.slurm`
- `data/fb15k-237/metadata.json`

If any value here conflicts with future reruns, treat JSON result artifacts in `results/slurm/` as the final authority.

---

## 2) Research Objective (PrimeKG + FB15k-237)

Evaluate whether lexical/semantic relation information improves KG link prediction across:

- Datasets: PrimeKG, FB15k-237
- Splits: edge-disjoint and node-disjoint
- Method families:
  - `baseline_gcn`
  - `baseline_sage`
  - `edge_aware_sage` (`concat` relation mode with OpenAI/Gemini/E5 relation embeddings)
  - `node2vec`
  - `hgt` (typed heterogeneous baseline)

Core question for paper narrative:

- How much of the gain is from relation-aware message passing itself?
- How sensitive are gains to relation embedding backend (OpenAI vs Gemini vs E5)?
- Do gains hold under the harder node-disjoint protocol?

---

## 3) Dataset Facts for Methods Section

### PrimeKG (repo-formatted file)

- File: `data/primekg.tsv`
- Total triples (rows): **8,100,498**
- Unique entities (source/target union): **129,262**
- Unique relations: **30**
- Top relation frequencies (by count):
  - `anatomy_protein_present`: 3,036,406
  - `drug_drug`: 2,672,628
  - `protein_protein`: 642,150
  - `disease_phenotype_positive`: 300,634
  - `bioprocess_protein`: 289,610

### FB15k-237

- Raw source repo: `KGraph/FB15k-237` (from `data/fb15k-237/metadata.json`)
- Split counts:
  - train: 272,115
  - valid: 17,535
  - test: 20,466
- Combined TSV file used by training pipeline: `data/fb15k-237/fb15k-237.tsv`
- Total triples (combined): **310,116**
- Unique entities: **14,541**
- Unique relations: **237**

---

## 4) Experimental Protocol and Fixed Settings

From the Slurm sweep scripts and runner defaults used for this campaign:

- Epochs: **100**
- Seed: **42** (primary sweep seed)
- Decoder: **dot**
- Negatives per positive (eval): **20**
- Split protocols evaluated: `edge`, `node`
- Prepared cache prebuild: required for sweep
- GPU required (`--require-cuda`)

### Prepared cache naming convention

- `cache/{dataset}-prepared_{split}_s{SEED}_d{IN_DIM}.pkl`
- Used values in this campaign for PrimeKG/FB15k sweep:
  - `SEED=42`
  - `IN_DIM=128` in the full array script

### Edge-aware SAGE setup in sweep

- Method: `edge_aware_sage`
- Relation mode: `concat`
- Relation embedding backends: `openai`, `gemini`, `e5`
- Semantic caches must already exist:
  - `cache/primekg-relations-{openai,gemini,e5}.pt`
  - `cache/fb15k237-relations-{openai,gemini,e5}.pt`

---

## 5) Run Matrix and Completion Status

Readiness summary from `results_primekg_fb15k237.md`:

- Successful result files included there: **42**
- Unique `(dataset, method, split)` combos tracked there: **20**
- Matrix status:
  - PrimeKG: `baseline_gcn`, `baseline_sage`, `edge_aware_sage`, `hgt`, `node2vec` all complete on edge + node
  - FB15k-237: `baseline_gcn`, `baseline_sage`, `edge_aware_sage`, `edge_aware_sage_node_emb`, `node2vec` all complete on edge + node
  - HGT for FB15k-237 is also present in `results/slurm` (`hgt-fb15k237-...` artifacts)

---

## 6) Main Results Tables (publication-ready values)

## PrimeKG (latest per method/split)

| Split | Method | ROC-AUC | AP | Hits@1 | Hits@3 | Hits@10 | Result File |
|---|---|---:|---:|---:|---:|---:|---|
| edge | baseline_gcn | 0.684083 | 0.085162 | 0.112812 | 0.295530 | 0.752625 | `results/slurm/primekg_edge_baseline_gcn_3249660_1.json` |
| edge | baseline_sage | 0.906463 | 0.243943 | 0.346790 | 0.737104 | 0.970119 | `results/slurm/primekg_edge_baseline_sage_3249660_2.json` |
| edge | edge_aware_sage | 0.946417 | 0.510746 | 0.553304 | 0.866961 | 0.984556 | `results/slurm/primekg_edge_edge_aware_sage_concat_openai_3252764_3.json` |
| edge | hgt | 0.971591 | 0.562997 | 0.883369 | 0.952484 | 0.956803 | `results/slurm/hgt-primekg-edge-s42-3256271_0.json` |
| edge | node2vec | 0.957959 | 0.651239 | 0.757166 | 0.901361 | 0.981177 | `results/slurm/primekg_edge_node2vec_3253102_0.json` |
| node | baseline_gcn | 0.679264 | 0.083839 | 0.103378 | 0.286331 | 0.764123 | `results/slurm/primekg_node_baseline_gcn_3249660_7.json` |
| node | baseline_sage | 0.878970 | 0.219399 | 0.287585 | 0.669110 | 0.947068 | `results/slurm/primekg_node_baseline_sage_3249660_8.json` |
| node | edge_aware_sage | 0.938104 | 0.473443 | 0.486559 | 0.828789 | 0.975292 | `results/slurm/primekg_node_edge_aware_sage_concat_openai_3252764_9.json` |
| node | hgt | 0.927521 | 0.653825 | 0.730331 | 0.899068 | 0.963251 | `results/slurm/hgt-primekg-node-s42-3256272_1.json` |

Note: `node2vec` node-split artifacts exist, but node-split node2vec is not directly inductive-comparable (transductive caveat in `EXPERIMENTS.md`).

## FB15k-237 (latest per method/split)

| Split | Method | ROC-AUC | AP | Hits@1 | Hits@3 | Hits@10 | Result File |
|---|---|---:|---:|---:|---:|---:|---|
| edge | baseline_gcn | 0.622956 | 0.089036 | 0.144163 | 0.306321 | 0.645194 | `results/slurm/fb15k237_edge_baseline_gcn_3249660_13.json` |
| edge | baseline_sage | 0.845738 | 0.282958 | 0.343605 | 0.627118 | 0.899834 | `results/slurm/fb15k237_edge_baseline_sage_3249660_14.json` |
| edge | edge_aware_sage | 0.895153 | 0.382479 | 0.451656 | 0.744591 | 0.949614 | `results/slurm/fb15k237_edge_edge_aware_sage_concat_openai_3252764_15.json` |
| edge | edge_aware_sage_node_emb | 0.881460 | 0.355993 | 0.421489 | 0.710178 | 0.934005 | `results/slurm/fb15k237_edge_edge_aware_sage_node_emb_e5_20260418_111425.json` |
| edge | hgt | 0.905330 | 0.297332 | 0.399460 | 0.824561 | 0.968961 | `results/slurm/hgt-fb15k237-edge-s42-3256272_2.json` |
| edge | node2vec | 0.806729 | 0.130731 | 0.346718 | 0.628857 | 0.907032 | `results/slurm/fb15k237_edge_node2vec_3253102_12.json` |
| node | baseline_gcn | 0.558025 | 0.058039 | 0.090986 | 0.219247 | 0.572410 | `results/slurm/fb15k237_node_baseline_gcn_3249660_19.json` |
| node | baseline_sage | 0.804065 | 0.199157 | 0.266413 | 0.542129 | 0.861889 | `results/slurm/fb15k237_node_baseline_sage_3249660_20.json` |
| node | edge_aware_sage | 0.880560 | 0.334111 | 0.408659 | 0.704036 | 0.932140 | `results/slurm/fb15k237_node_edge_aware_sage_concat_openai_3252764_21.json` |
| node | edge_aware_sage_node_emb | 0.870008 | 0.300667 | 0.371222 | 0.668261 | 0.928857 | `results/slurm/fb15k237_node_edge_aware_sage_node_emb_e5_20260418_111425.json` |
| node | hgt | 0.830096 | 0.317387 | 0.409258 | 0.747245 | 0.811903 | `results/slurm/hgt-fb15k237-node-s42-3256272_3.json` |

---

## 7) Embedding Backend Ablation (Edge-Aware SAGE, concat mode)

## PrimeKG

| Split | Embedding | ROC-AUC | AP | Hits@1 | Hits@3 | Hits@10 |
|---|---|---:|---:|---:|---:|---:|
| edge | openai | 0.946417 | 0.510746 | 0.553304 | 0.866961 | 0.984556 |
| edge | gemini | 0.945611 | 0.504044 | 0.543720 | 0.863530 | 0.983262 |
| edge | e5 | 0.946804 | 0.508607 | 0.547940 | 0.865827 | 0.984785 |
| node | openai | 0.938104 | 0.473443 | 0.486559 | 0.828789 | 0.975292 |
| node | gemini | 0.937565 | 0.477573 | 0.496542 | 0.833584 | 0.973711 |
| node | e5 | 0.936724 | 0.469222 | 0.491130 | 0.831240 | 0.972802 |

## FB15k-237

| Split | Embedding | ROC-AUC | AP | Hits@1 | Hits@3 | Hits@10 |
|---|---|---:|---:|---:|---:|---:|
| edge | openai | 0.895153 | 0.382479 | 0.451656 | 0.744591 | 0.949614 |
| edge | gemini | 0.896220 | 0.392829 | 0.456347 | 0.746088 | 0.949412 |
| edge | e5 | 0.891156 | 0.366520 | 0.442638 | 0.735048 | 0.945206 |
| node | openai | 0.880560 | 0.334111 | 0.408659 | 0.704036 | 0.932140 |
| node | gemini | 0.876454 | 0.329566 | 0.402416 | 0.695259 | 0.927093 |
| node | e5 | 0.871743 | 0.323292 | 0.398220 | 0.690455 | 0.919188 |

Interpretation for writer:

- Gains from edge-aware semantic relation injection are robust across embedding providers.
- Provider ranking is dataset/split dependent (no single provider dominates all settings).
- Edge-aware > baseline_sage across both datasets and both split protocols on all main ranking metrics.

---

## 8) Candidate Paper Claims (supported by current artifacts)

1. Relation-aware message passing with lexical semantic relation initialization substantially improves GraphSAGE over relation-agnostic baselines on PrimeKG and FB15k-237.
2. Improvements persist under node-disjoint evaluation, supporting stronger OOD-like generalization claims versus edge-only split claims.
3. Embedding backend choice (OpenAI/Gemini/E5) changes absolute values modestly but does not change the overall conclusion that relation-aware semantic conditioning helps.
4. A lightweight edge-aware GraphSAGE variant can be competitive with stronger heterogeneous baselines depending on metric/split; comparisons should be metric-specific (ROC-AUC vs AP vs Hits@K).

Conservative framing notes:

- Keep node2vec as a transductive reference; do not overclaim node-split inductive fairness.
- Separate `edge_aware_sage_node_emb` from main edge-only semantic method because it changes feature assumptions.

---

## 9) Exact Reproduction Commands (campaign backbone)

### A) Build prepared caches first (CPU)

```bash
sbatch scripts/slurm/primekg_fb15k_build_prepared_caches.slurm
```

This builds:

- `cache/primekg-prepared_edge_s42_d128.pkl`
- `cache/primekg-prepared_node_s42_d128.pkl`
- `cache/fb15k237-prepared_edge_s42_d128.pkl`
- `cache/fb15k237-prepared_node_s42_d128.pkl`

### B) Main GPU sweep (24-task array)

```bash
sbatch scripts/slurm/primekg_fb15k_full_sweep_array.slurm
```

Task matrix:

- datasets: `primekg`, `fb15k237`
- splits: `edge`, `node`
- methods:
  - `node2vec`
  - `baseline_gcn`
  - `baseline_sage`
  - `edge_aware_sage_concat_openai`
  - `edge_aware_sage_concat_gemini`
  - `edge_aware_sage_concat_e5`

### C) HGT sweep (4-task array)

```bash
sbatch scripts/slurm/hgt_primekg_fb15k_split_array.slurm
```

Important:

- Set `HGT_RELATION_FB15K237` to a valid FB15k-237 predicate before submit.
- Neighbor-sampling backend is required by script guard.

---

## 10) Artifact Manifest for Paper Writing

## Data and metadata

- `data/primekg.tsv`
- `data/fb15k-237/fb15k-237.tsv`
- `data/fb15k-237/metadata.json`

## Relation caches

- `cache/primekg-relations-openai.pt`
- `cache/primekg-relations-gemini.pt`
- `cache/primekg-relations-e5.pt`
- `cache/fb15k237-relations-openai.pt`
- `cache/fb15k237-relations-gemini.pt`
- `cache/fb15k237-relations-e5.pt`

## Prepared split caches

- `cache/primekg-prepared_edge_s42_d128.pkl`
- `cache/primekg-prepared_node_s42_d128.pkl`
- `cache/fb15k237-prepared_edge_s42_d128.pkl`
- `cache/fb15k237-prepared_node_s42_d128.pkl`

## Result JSONs (PrimeKG/FB15k/HGT campaign)

All current relevant outputs are under:

- `results/slurm/primekg_*.json`
- `results/slurm/fb15k237_*.json`
- `results/slurm/hgt-primekg-*.json`
- `results/slurm/hgt-fb15k237-*.json`

Current count from repository snapshot: **53** JSON files matching those patterns.

## Sweep logs

- `logs/slurm/kgml-primekg-fb15k-sweep-*.out/.err`
- `logs/slurm/run-primekg-*.log`
- `logs/slurm/run-fb15k237-*.log`
- `logs/slurm/kgml-hgt-primekg-fb15k-*.out/.err`

---

## 11) Writer Checklist by Paper Section

### Abstract

- Use the cross-dataset, cross-split improvement story:
  - PrimeKG and FB15k-237
  - edge + node split consistency
  - relation-aware semantic injection with low architectural overhead

### Methods

- Use `run_gpu_method` pipeline details from `EXPERIMENTS.md`
- Describe split protocols and negative sampling defaults
- Describe edge-aware `concat` mechanism and relation embedding caches

### Experimental Setup

- Include seed, epochs, decoder, negatives-per-pos, and prepared-cache strategy
- Explicitly list methods and embedding backends
- Mention hardware via Slurm logs if you want system details in appendix

### Results

- Main tables from Section 6
- Embedding ablation from Section 7
- Optional appendix: include `edge_aware_sage_node_emb` as supplementary variant

### Discussion

- Address fairness caveats (node2vec transductive nature)
- Explain backend sensitivity as second-order effect
- Position relation-aware message passing as dominant first-order effect

---

## 12) Known Caveats / Open Decisions Before Submission

1. Decide final primary metric hierarchy (AP vs ROC-AUC vs Hits@10).  
   Some method ordering changes by metric.
2. Decide whether to include HGT in main table or keep it as secondary baseline with nuance by split.
3. Decide whether to include `edge_aware_sage_node_emb` in main text; currently cleaner to keep as appendix ablation.
4. If required by venue, aggregate multi-seed confidence intervals (current campaign is strongly centered on seed 42 sweep artifacts).

---

## 13) One-Line Takeaway

Across PrimeKG and FB15k-237, relation-aware semantic edge conditioning consistently strengthens GraphSAGE in both edge-disjoint and node-disjoint link prediction, with robust gains across OpenAI/Gemini/E5 relation embeddings and a reproducible Slurm-backed pipeline.

---

## 14) Manuscript-Ready Methods (detailed draft block)

Use or adapt the following directly as the Methods section in the paper.

### 14.1 Task definition

We study binary link prediction in multi-relational knowledge graphs. Given a graph \(G=(V,E)\) where each observed edge is a triple \((u,r,v)\) with relation type \(r \in \mathcal{R}\), the model outputs a score \(s(u,r,v)\) estimating whether the triple is valid. We evaluate rank-based and threshold-free metrics (Hits@K, AP, ROC-AUC) under two split regimes.

### 14.2 Datasets

Two benchmarks are used in a unified pipeline:

- PrimeKG (`data/primekg.tsv`): 8,100,498 triples, 129,262 unique entities, 30 relations.
- FB15k-237 (`data/fb15k-237/fb15k-237.tsv`): 310,116 triples (train 272,115; valid 17,535; test 20,466), 14,541 entities, 237 relations.

All methods consume the same TSV edge-list format in this campaign.

### 14.3 Split protocols

We report both:

1. **Edge-disjoint split (`split_protocol=edge`)**: entities can appear in train/val/test; only positive edges are held out.
2. **Node-disjoint split (`split_protocol=node`)**: held-out nodes are excluded from training positives; this is a stricter, OOD-like setting for entity generalization.

The node split is treated as the stronger generalization test in claims.

### 14.4 Negative sampling and evaluation setup

- Validation/test negatives per positive: 20 (`--negatives-per-pos 20`).
- Decoder: dot-product (`--decoder dot`) in this sweep.
- Metrics extracted from result JSONs:
  - ROC-AUC
  - Average Precision (AP)
  - Hits@1, Hits@3, Hits@10

### 14.5 Models compared

Primary methods:

- `baseline_gcn`: homogeneous GCN baseline.
- `baseline_sage`: relation-agnostic GraphSAGE baseline.
- `edge_aware_sage`: relation-aware GraphSAGE with lexical-semantic relation vectors.
- `node2vec`: transductive embedding baseline.
- `hgt`: Heterogeneous Graph Transformer baseline (`run_hgt` pipeline).

### 14.6 Edge-aware semantic mechanism (implementation-aligned)

For `edge_aware_sage` in this campaign:

- Relation control mode: `concat`.
- Relation embeddings are loaded from frozen cache files (`.pt`), one vector per relation type.
- Backends compared:
  - OpenAI
  - Gemini
  - E5
- The model learns task-specific transformations during message passing while relation vectors initialize semantic relation context.

In the sweep scripts, semantic runs use:

- `--method edge_aware_sage`
- `--semantic --strict-semantic`
- `--embedding-model {openai|gemini|e5}`
- `--edge-relation-mode concat`
- `--edge-dim 128`
- dataset-specific `--semantic-cache`

### 14.7 Training protocol and controls

From the main PrimeKG/FB15k campaign scripts:

- Seed: 42
- Epochs: 100
- Input dimension: 128 for prepared caches used in sweep
- CUDA required (`--require-cuda`)
- Prepared split caches are precomputed and reused to keep data splits identical across methods

This prebuild-and-reuse approach is critical for fair model comparison because it removes split drift across runs.

### 14.8 Reproducibility workflow

1. Build prepared caches (CPU):
   - `sbatch scripts/slurm/primekg_fb15k_build_prepared_caches.slurm`
2. Run 24-task model sweep:
   - `sbatch scripts/slurm/primekg_fb15k_full_sweep_array.slurm`
3. Run 4-task HGT sweep:
   - `sbatch scripts/slurm/hgt_primekg_fb15k_split_array.slurm`
4. Collect outputs from `results/slurm/*.json`.

### 14.9 Fairness notes (must be stated in paper)

- Node2Vec is transductive; it is informative on edge split but should be interpreted cautiously for node-disjoint generalization claims.
- `edge_aware_sage_node_emb` is a variant with additional node embedding inputs and is not the cleanest apples-to-apples comparator to edge-only semantic injection.
- Metric ranking can differ by metric family (e.g., AP vs ROC-AUC), so claims should specify the target metric.

---

## 15) Manuscript-Ready Results (detailed draft block)

Use this section as writing substrate; all values are already in Section 6/7 tables.

### 15.1 Main trend

Across both datasets and both split protocols, `edge_aware_sage` consistently outperforms `baseline_sage` and `baseline_gcn` by substantial margins on ROC-AUC, AP, and Hits@K. This indicates that relation-aware semantic conditioning improves both ranking quality and top-k retrieval quality.

### 15.2 PrimeKG

On PrimeKG edge split:

- `baseline_sage`: ROC-AUC 0.906, AP 0.244
- `edge_aware_sage` (OpenAI cache): ROC-AUC 0.946, AP 0.511

Absolute gains versus `baseline_sage`:

- ROC-AUC: +0.040
- AP: +0.267

On PrimeKG node split:

- `baseline_sage`: ROC-AUC 0.879, AP 0.219
- `edge_aware_sage` (OpenAI cache): ROC-AUC 0.938, AP 0.473

Absolute gains:

- ROC-AUC: +0.059
- AP: +0.254

Interpretation: the edge-aware semantic model retains strong gains under stricter node-disjoint evaluation, supporting stronger inductive/OOD-style claims.

### 15.3 FB15k-237

On FB15k-237 edge split:

- `baseline_sage`: ROC-AUC 0.846, AP 0.283
- `edge_aware_sage` (OpenAI cache): ROC-AUC 0.895, AP 0.382

Absolute gains:

- ROC-AUC: +0.049
- AP: +0.100

On FB15k-237 node split:

- `baseline_sage`: ROC-AUC 0.804, AP 0.199
- `edge_aware_sage` (OpenAI cache): ROC-AUC 0.881, AP 0.334

Absolute gains:

- ROC-AUC: +0.076
- AP: +0.135

Interpretation: improvements transfer to high-relation-cardinality benchmark conditions and are not dataset-specific to biomedical graph structure.

### 15.4 Embedding backend ablation

Provider choice affects absolute values modestly but does not change the top-level conclusion:

- PrimeKG edge:
  - OpenAI AP 0.511
  - Gemini AP 0.504
  - E5 AP 0.509
- PrimeKG node:
  - OpenAI AP 0.473
  - Gemini AP 0.478
  - E5 AP 0.469
- FB15k edge:
  - OpenAI AP 0.382
  - Gemini AP 0.393
  - E5 AP 0.367
- FB15k node:
  - OpenAI AP 0.334
  - Gemini AP 0.330
  - E5 AP 0.323

Conclusion: lexical-semantic relation conditioning is robust to embedding backend; backend tuning is a second-order optimization.

### 15.5 Relation to HGT and Node2Vec

- HGT is competitive and sometimes stronger on selected metrics/splits.
- Node2Vec is very strong in transductive edge settings on PrimeKG but should not be overused for inductive node-split claims.
- A clean narrative is:
  - `edge_aware_sage` delivers reliable improvements over relation-agnostic GNN baselines with lower methodological complexity.
  - HGT provides an additional high-capacity heterogeneous reference point.

---

## 16) Manuscript-Ready Discussion (detailed draft block)

### 16.1 Core interpretation

The experiments support that relation-aware semantic context is a high-leverage inductive bias for KG link prediction. Improvements are stable across biomedical and general-domain KGs, and remain under node-disjoint splitting, suggesting gains are not merely memorization over seen entities.

### 16.2 Why this matters

- Practicality: the method plugs into a standard GraphSAGE workflow.
- Scalability: prepared-cache + array-sweep pipeline makes large benchmark runs operationally tractable.
- Robustness: semantic backend changes do not collapse performance, reducing dependence on one proprietary embedding source.

### 16.3 Limits and caveats

1. **Seed concentration**: current principal comparison is centered on seed 42; multi-seed uncertainty bands should be added for camera-ready rigor.
2. **Metric sensitivity**: model ordering can change between ROC-AUC/AP/Hits@K; final claims should specify primary metric.
3. **Comparator scope**: `edge_aware_sage_node_emb` is not fully apples-to-apples with edge-only semantic injection.
4. **Transductive comparator caveat**: Node2Vec should remain a contextual baseline, not core evidence for node-split generalization.

### 16.4 Recommended additional analyses before final submission

- 3-5 seed reruns for key rows (baseline_sage vs edge_aware_sage on both datasets and both splits).
- Statistical confidence reporting (mean +/- std or bootstrap CI for AP and ROC-AUC).
- Wall-clock and GPU-hour efficiency table for methods (especially vs HGT).
- Optional calibration or threshold analysis if deployment use-case is emphasized.

### 16.5 Safe claim language for final paper

Preferred:

- "consistently improves over relation-agnostic GraphSAGE under both split protocols"
- "robust across three relation embedding backends"
- "competitive with stronger heterogeneous baselines on multiple settings"

Avoid without multi-seed confirmation:

- "state-of-the-art"
- "uniformly best across all metrics and splits"
- "statistically significant" (unless formal test is added)

---

## 17) Fill-in Template Text (copy/paste helper)

### Methods paragraph starter

"We evaluate link prediction on PrimeKG and FB15k-237 using a unified training pipeline. We compare relation-agnostic baselines (GCN, GraphSAGE), a relation-aware semantic GraphSAGE variant, Node2Vec, and HGT under both edge-disjoint and node-disjoint protocols. For all primary runs, we fix seed=42, epochs=100, dot-product decoding, and 20 negatives per positive during evaluation. To ensure strict fairness, we precompute and reuse split caches per dataset and split."

### Results paragraph starter

"Across both datasets and both split protocols, relation-aware semantic GraphSAGE consistently improves over standard GraphSAGE. On PrimeKG, AP improves from 0.244 to 0.511 (edge split) and from 0.219 to 0.473 (node split). On FB15k-237, AP improves from 0.283 to 0.382 (edge split) and from 0.199 to 0.334 (node split). Similar trends are observed for ROC-AUC and Hits@K."

### Discussion paragraph starter

"These results suggest that lexical semantic relation conditioning provides a practical and robust inductive bias for multi-relational link prediction. The gains hold in node-disjoint evaluation, indicating improved generalization beyond transductive edge reconstruction. While embedding backend choice affects absolute values, the improvement trend is stable across OpenAI, Gemini, and E5."

