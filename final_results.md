# New Decoder Results

This summarizes the completed runs from the decoder-era experiments. I included only the jobs with final metrics available in `results/slurm/*.json`.

## PrimeKG

| Split | Method | ROC AUC | AP | Hits@1 | Hits@3 | Hits@10 |
|---|---|---:|---:|---:|---:|---:|
| Edge | `rotate` | 0.97498 | 0.54505 | 0.84048 | 0.96899 | 0.99220 |
| Edge | `node2vec` | 0.95796 | 0.65124 | 0.75717 | 0.90136 | 0.98118 |
| Edge | `edge_aware_sage_concat` openai | 0.94677 | 0.51179 | 0.55846 | 0.86781 | 0.98305 |
| Edge | `edge_aware_sage_concat` gemini | 0.94648 | 0.50942 | 0.54661 | 0.86395 | 0.98339 |
| Edge | `edge_aware_sage_concat` e5 | 0.94533 | 0.50560 | 0.55024 | 0.86425 | 0.98262 |
| Edge | `baseline_sage` | 0.90917 | 0.24486 | 0.34301 | 0.74666 | 0.96956 |
| Edge | `baseline_gcn` | 0.69921 | 0.09311 | 0.12614 | 0.32187 | 0.77333 |
| Node | `edge_aware_sage_concat` openai | 0.93810 | 0.47344 | 0.48656 | 0.82879 | 0.97529 |
| Node | `edge_aware_sage_concat` gemini | 0.93757 | 0.47757 | 0.49654 | 0.83358 | 0.97371 |
| Node | `edge_aware_sage_concat` e5 | 0.93672 | 0.46922 | 0.49113 | 0.83124 | 0.97280 |
| Node | `baseline_sage` | 0.87897 | 0.21940 | 0.28758 | 0.66911 | 0.94707 |
| Node | `baseline_gcn` | 0.67926 | 0.08384 | 0.10338 | 0.28633 | 0.76412 |
| Node | `rotate` | 0.49605 | 0.04681 | 0.04245 | 0.13498 | 0.47511 |

## FB15k-237

| Split | Method | ROC AUC | AP | Hits@1 | Hits@3 | Hits@10 |
|---|---|---:|---:|---:|---:|---:|
| Edge | `edge_aware_sage_concat` openai | 0.89762 | 0.38953 | 0.45720 | 0.74649 | 0.95176 |
| Edge | `edge_aware_sage_concat` gemini | 0.89576 | 0.39456 | 0.46084 | 0.74916 | 0.95046 |
| Edge | `edge_aware_sage_concat` e5 | 0.89295 | 0.38026 | 0.44963 | 0.73828 | 0.94589 |
| Edge | `edge_aware_sage_node_emb` | 0.89005 | 0.37833 | 0.43394 | 0.71984 | 0.94351 |
| Edge | `edge_aware_sage_node_emb` e5 | 0.88146 | 0.35599 | 0.42149 | 0.71018 | 0.93400 |
| Edge | `baseline_sage` | 0.85089 | 0.28887 | 0.35165 | 0.63917 | 0.90618 |
| Edge | `node2vec` | 0.76841 | 0.11387 | 0.26612 | 0.52792 | 0.85936 |
| Edge | `rotate` | 0.71986 | 0.17378 | 0.22718 | 0.42533 | 0.75830 |
| Edge | `baseline_gcn` | 0.62732 | 0.08653 | 0.14449 | 0.30288 | 0.65724 |
| Node | `edge_aware_sage_concat` openai | 0.88056 | 0.33411 | 0.40866 | 0.70404 | 0.93214 |
| Node | `edge_aware_sage_concat` gemini | 0.87645 | 0.32957 | 0.40242 | 0.69526 | 0.92709 |
| Node | `edge_aware_sage_concat` e5 | 0.87174 | 0.32329 | 0.39822 | 0.69046 | 0.91919 |
| Node | `edge_aware_sage_node_emb` | 0.87406 | 0.31267 | 0.38831 | 0.68188 | 0.92985 |
| Node | `edge_aware_sage_node_emb` e5 | 0.87001 | 0.30067 | 0.37122 | 0.66826 | 0.92886 |
| Node | `baseline_sage` | 0.80407 | 0.19916 | 0.26641 | 0.54213 | 0.86189 |
| Node | `baseline_gcn` | 0.55803 | 0.05804 | 0.09099 | 0.21925 | 0.57241 |
| Node | `rotate` | 0.50070 | 0.04768 | 0.04743 | 0.14585 | 0.47528 |

## Notes

- `rotate` is the strongest overall on PrimeKG, especially on the edge split.
- On FB15k-237, `edge_aware_sage_concat` with OpenAI embeddings is the strongest among the graph-encoder methods we completed.
- `edge_aware_sage_node_emb` is competitive on FB15k-237 but slightly below the best `edge_aware_sage_concat` results.
- `node2vec` is strong on PrimeKG edge split, but its node-split JSON artifacts in this tree did not contain final metrics, so I excluded those from this summary.

