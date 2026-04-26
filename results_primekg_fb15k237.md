# PrimeKG + FB15K-237 Results Summary

This file summarizes successful result JSON files in `results/slurm` and adds readiness status for required edge/node runs per method.

- Successful result files included: **42**
- Unique (dataset, method, split) combos: **20**

## Readiness Matrix

| Dataset | Method | Edge Split | Node Split | Good To Go | Notes |
|---|---|---|---|---|---|
| primekg | baseline_gcn | complete | complete | yes |  |
| primekg | baseline_sage | complete | complete | yes |  |
| primekg | edge_aware_sage | complete | complete | yes |  |
| primekg | hgt | complete | complete | yes |  |
| primekg | node2vec | complete | complete | yes |  |
| fb15k237 | baseline_gcn | complete | complete | yes |  |
| fb15k237 | baseline_sage | complete | complete | yes |  |
| fb15k237 | edge_aware_sage | complete | complete | yes |  |
| fb15k237 | edge_aware_sage_node_emb | complete | complete | yes |  |
| fb15k237 | node2vec | complete | complete | yes |  |

## primekg (latest per method/split)

| Split | Method | ROC-AUC | AP | Hits@1 | Hits@3 | Hits@10 | Neighbor Sampling | Sampler Backend | Epochs | Result File |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---|
| edge | baseline_gcn | 0.684083 | 0.085162 | 0.112812 | 0.295530 | 0.752625 | N/A | N/A | 100 | `results/slurm/primekg_edge_baseline_gcn_3249660_1.json` |
| edge | baseline_sage | 0.906463 | 0.243943 | 0.346790 | 0.737104 | 0.970119 | N/A | N/A | 100 | `results/slurm/primekg_edge_baseline_sage_3249660_2.json` |
| edge | edge_aware_sage | 0.946417 | 0.510746 | 0.553304 | 0.866961 | 0.984556 | N/A | N/A | 100 | `results/slurm/primekg_edge_edge_aware_sage_concat_openai_3252764_3.json` |
| edge | hgt | 0.971591 | 0.562997 | 0.883369 | 0.952484 | 0.956803 | True | True | N/A | `results/slurm/hgt-primekg-edge-s42-3256271_0.json` |
| edge | node2vec | 0.957959 | 0.651239 | 0.757166 | 0.901361 | 0.981177 | N/A | N/A | 100 | `results/slurm/primekg_edge_node2vec_3253102_0.json` |
| node | baseline_gcn | 0.679264 | 0.083839 | 0.103378 | 0.286331 | 0.764123 | N/A | N/A | 100 | `results/slurm/primekg_node_baseline_gcn_3249660_7.json` |
| node | baseline_sage | 0.878970 | 0.219399 | 0.287585 | 0.669110 | 0.947068 | N/A | N/A | 100 | `results/slurm/primekg_node_baseline_sage_3249660_8.json` |
| node | edge_aware_sage | 0.938104 | 0.473443 | 0.486559 | 0.828789 | 0.975292 | N/A | N/A | 100 | `results/slurm/primekg_node_edge_aware_sage_concat_openai_3252764_9.json` |
| node | hgt | 0.927521 | 0.653825 | 0.730331 | 0.899068 | 0.963251 | True | True | N/A | `results/slurm/hgt-primekg-node-s42-3256272_1.json` |
| node | node2vec | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 100 | `results/slurm/primekg_node_node2vec_3253102_6.json` |

### primekg edge-aware SAGE embedding breakdown

| Split | Embedding Type | ROC-AUC | AP | Hits@1 | Hits@3 | Hits@10 | Result File |
|---|---|---:|---:|---:|---:|---:|---|
| edge | openai | 0.946417 | 0.510746 | 0.553304 | 0.866961 | 0.984556 | `results/slurm/primekg_edge_edge_aware_sage_concat_openai_3252764_3.json` |
| edge | gemini | 0.945611 | 0.504044 | 0.543720 | 0.863530 | 0.983262 | `results/slurm/primekg_edge_edge_aware_sage_concat_gemini_3252847_4.json` |
| edge | e5 | 0.946804 | 0.508607 | 0.547940 | 0.865827 | 0.984785 | `results/slurm/primekg_edge_edge_aware_sage_concat_e5_3252764_5.json` |
| node | openai | 0.938104 | 0.473443 | 0.486559 | 0.828789 | 0.975292 | `results/slurm/primekg_node_edge_aware_sage_concat_openai_3252764_9.json` |
| node | gemini | 0.937565 | 0.477573 | 0.496542 | 0.833584 | 0.973711 | `results/slurm/primekg_node_edge_aware_sage_concat_gemini_3252847_10.json` |
| node | e5 | 0.936724 | 0.469222 | 0.491130 | 0.831240 | 0.972802 | `results/slurm/primekg_node_edge_aware_sage_concat_e5_3252764_11.json` |

## fb15k237 (latest per method/split)

| Split | Method | ROC-AUC | AP | Hits@1 | Hits@3 | Hits@10 | Neighbor Sampling | Sampler Backend | Epochs | Result File |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---|
| edge | baseline_gcn | 0.622956 | 0.089036 | 0.144163 | 0.306321 | 0.645194 | N/A | N/A | 100 | `results/slurm/fb15k237_edge_baseline_gcn_3249660_13.json` |
| edge | baseline_sage | 0.845738 | 0.282958 | 0.343605 | 0.627118 | 0.899834 | N/A | N/A | 100 | `results/slurm/fb15k237_edge_baseline_sage_3249660_14.json` |
| edge | edge_aware_sage | 0.895153 | 0.382479 | 0.451656 | 0.744591 | 0.949614 | N/A | N/A | 100 | `results/slurm/fb15k237_edge_edge_aware_sage_concat_openai_3252764_15.json` |
| edge | edge_aware_sage_node_emb | 0.881460 | 0.355993 | 0.421489 | 0.710178 | 0.934005 | N/A | N/A | 100 | `results/slurm/fb15k237_edge_edge_aware_sage_node_emb_e5_20260418_111425.json` |
| edge | hgt | 0.905330 | 0.297332 | 0.399460 | 0.824561 | 0.968961 | True | True | N/A | `results/slurm/hgt-fb15k237-edge-s42-3256272_2.json` |
| edge | node2vec | 0.806729 | 0.130731 | 0.346718 | 0.628857 | 0.907032 | N/A | N/A | 100 | `results/slurm/fb15k237_edge_node2vec_3253102_12.json` |
| node | baseline_gcn | 0.558025 | 0.058039 | 0.090986 | 0.219247 | 0.572410 | N/A | N/A | 100 | `results/slurm/fb15k237_node_baseline_gcn_3249660_19.json` |
| node | baseline_sage | 0.804065 | 0.199157 | 0.266413 | 0.542129 | 0.861889 | N/A | N/A | 100 | `results/slurm/fb15k237_node_baseline_sage_3249660_20.json` |
| node | edge_aware_sage | 0.880560 | 0.334111 | 0.408659 | 0.704036 | 0.932140 | N/A | N/A | 100 | `results/slurm/fb15k237_node_edge_aware_sage_concat_openai_3252764_21.json` |
| node | edge_aware_sage_node_emb | 0.870008 | 0.300667 | 0.371222 | 0.668261 | 0.928857 | N/A | N/A | 100 | `results/slurm/fb15k237_node_edge_aware_sage_node_emb_e5_20260418_111425.json` |
| node | hgt | 0.830096 | 0.317387 | 0.409258 | 0.747245 | 0.811903 | True | True | N/A | `results/slurm/hgt-fb15k237-node-s42-3256272_3.json` |
| node | node2vec | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 100 | `results/slurm/fb15k237_node_node2vec_3253102_18.json` |

### fb15k237 edge-aware SAGE embedding breakdown

| Split | Embedding Type | ROC-AUC | AP | Hits@1 | Hits@3 | Hits@10 | Result File |
|---|---|---:|---:|---:|---:|---:|---|
| edge | openai | 0.895153 | 0.382479 | 0.451656 | 0.744591 | 0.949614 | `results/slurm/fb15k237_edge_edge_aware_sage_concat_openai_3252764_15.json` |
| edge | gemini | 0.896220 | 0.392829 | 0.456347 | 0.746088 | 0.949412 | `results/slurm/fb15k237_edge_edge_aware_sage_concat_gemini_3252847_16.json` |
| edge | e5 | 0.891156 | 0.366520 | 0.442638 | 0.735048 | 0.945206 | `results/slurm/fb15k237_edge_edge_aware_sage_concat_e5_3252764_17.json` |
| edge | openai_node_emb | 0.890050 | 0.378328 | 0.433944 | 0.719843 | 0.943508 | `results/slurm/fb15k237_edge_edge_aware_sage_node_emb_20260417_141445.json` |
| edge | e5_node_emb | 0.881460 | 0.355993 | 0.421489 | 0.710178 | 0.934005 | `results/slurm/fb15k237_edge_edge_aware_sage_node_emb_e5_20260418_111425.json` |
| node | openai | 0.880560 | 0.334111 | 0.408659 | 0.704036 | 0.932140 | `results/slurm/fb15k237_node_edge_aware_sage_concat_openai_3252764_21.json` |
| node | gemini | 0.876454 | 0.329566 | 0.402416 | 0.695259 | 0.927093 | `results/slurm/fb15k237_node_edge_aware_sage_concat_gemini_3252847_22.json` |
| node | e5 | 0.871743 | 0.323292 | 0.398220 | 0.690455 | 0.919188 | `results/slurm/fb15k237_node_edge_aware_sage_concat_e5_3252764_23.json` |
| node | openai_node_emb | 0.874057 | 0.312666 | 0.388309 | 0.681882 | 0.929850 | `results/slurm/fb15k237_node_edge_aware_sage_node_emb_20260417_141445.json` |
| node | e5_node_emb | 0.870008 | 0.300667 | 0.371222 | 0.668261 | 0.928857 | `results/slurm/fb15k237_node_edge_aware_sage_node_emb_e5_20260418_111425.json` |

### fb15k237 HGT (completed runs)

| Split | Method | ROC-AUC | AP | Hits@1 | Hits@3 | Hits@10 |
|---|---|---:|---:|---:|---:|---:|
| edge | hgt | 0.905330 | 0.297332 | 0.399460 | 0.824561 | 0.968961 |
| node | hgt | 0.830096 | 0.317387 | 0.409258 | 0.747245 | 0.811903 |
