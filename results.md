# Knowledge Graph Relation Embedding Experiments

**How to run experiments (setup, splits, CLIs, smoke tests):** see [EXPERIMENTS.md](EXPERIMENTS.md). **Maintainer/agent conventions (APIs, caches, caveats):** see [AGENTS.md](AGENTS.md).

This document tracks all experiments investigating why semantic relation embeddings don't help (and sometimes hurt) edge-aware GraphSAGE experiments on knowledge graphs.

## Goal

Investigate why semantic relation embeddings (OpenAI embeddings of relation glossary text) don't improve edge-aware GraphSAGE link prediction, and explore alternatives including:
1. Different embedding models (SapBERT vs OpenAI)
2. Different text modes (raw glossary vs canonical cleaned text)
3. Relation diversity in the dataset
4. Different edge relation mechanisms (concat, gated, basis_mixture, film)
5. Semantic alignment regularizer for basis_mixture

## Executive Summary

### Main Findings

1. **OpenAI embeddings don't capture semantic opposition** - treat opposing relations as similar (cosine ~0.8)
2. **SapBERT shows better opposition separation** - raw+sapbert achieves 0.07 average cosine on opposing pairs
3. **Canonical text mode hurts SapBERT** - cleaning the glossary removes useful structural cues
4. **All edge-aware methods beat baseline by ~6-7%** on high-diversity data
5. **The learned projection dominates** - semantic ≈ random ≈ shuffled in final performance
6. **Semantic alignment regularizer helps** basis_mixture only with semantic embeddings (+1.2% ROC)
7. **FiLM performs comparably to concat** - new mechanism works but doesn't improve over concat

### Best Results (25k edges, high diversity)

| Method | ROC-AUC | H@10 | Delta vs Baseline |
|--------|---------|------|-------------------|
| baseline_sage | 0.5404 | 0.5367 | — |
| concat_random | 0.6085 | 0.6047 | +6.8% |
| concat_sapbert | 0.6078 | 0.6051 | +6.7% |
| basis_no_align | 0.6046 | 0.6066 | +6.4% |
| film_sapbert | 0.6039 | 0.6093 | +6.4% |
| film_random | 0.6013 | 0.5954 | +6.1% |

---

## Full DRKG (`drkg.tsv`) — Slurm `drkg_full_experiments_array`

End-to-end link prediction on the **full** DRKG graph loaded from repo-root `drkg.tsv` (no `max_edges` cap). **Split:** `--split-protocol edge` (edge-disjoint train / val / test). **Training:** 100 epochs, `in_dim=64`, `decoder=dot`, `negatives_per_pos=20`, global negative sampling. **Edge-aware runs** use `--edge-relation-mode film`, SapBERT relation cache `cache/drkg-relations-sapbert.pt`, and `--embedding-model sapbert` unless noted.

Metrics below are taken from each run’s **`results/slurm/*.json`** (`val_metrics` / `test_metrics` aggregates). Bucketed metrics (low / medium / high relation-diversity) are inside the same JSON files.

### Slurm array 3236721 (2026-04-06)

Prepared-dataset cache **`cache/drkg-prepared_edge_s42.pkl`** and **seed 42** were used for this submission (faster startup vs rebuilding PyG data on every task). One row per `scripts/slurm/drkg_full_experiments_array.slurm` array task (0–3).

| Task | Method | Neighbor agg | Val ROC-AUC | Test ROC-AUC | Test AP | Test H@10 | GPU | Result file |
|------|--------|--------------|-------------|--------------|---------|-----------|-----|-------------|
| 0 | `baseline_sage` | mean | 0.8544 | 0.8548 | 0.2099 | 0.9428 | L40S | `results/slurm/drkg_full_bsage_mean_3236721_0.json` |
| 1 | `baseline_sage` | max (pool) | 0.8760 | 0.8757 | 0.2314 | 0.9564 | L40S | `results/slurm/drkg_full_bsage_pool_3236721_1.json` |
| 2 | `edge_aware_sage` (FiLM) | mean | 0.9212 | 0.9210 | 0.4246 | 0.9754 | H100 NVL | `results/slurm/drkg_full_easage_film_mean_3236721_2.json` |
| 3 | `edge_aware_sage` (FiLM) | max (pool) | 0.9246 | 0.9245 | 0.4170 | 0.9790 | H100 NVL | `results/slurm/drkg_full_easage_film_pool_3236721_3.json` |

**Slurm job IDs:** `3236721_0` … `3236721_3` (see `logs/slurm/drkg-full-3236721_*.err` and `logs/slurm/run-drkg-3236721_*.log`).

### Slurm array 3236475 (2026-04-05→06)

Same sweep template; **tasks 0–1** (baselines only) finished with JSON on disk. **Tasks 2–3** (edge-aware FiLM on P100) were still running later; **add two rows here** when `results/slurm/drkg_full_easage_film_mean_3236475_2.json` and `results/slurm/drkg_full_easage_film_pool_3236475_3.json` exist.

| Task | Method | Neighbor agg | Val ROC-AUC | Test ROC-AUC | Test AP | Test H@10 | GPU | Result file |
|------|--------|--------------|-------------|--------------|---------|-----------|-----|-------------|
| 0 | `baseline_sage` | mean | 0.8538 | 0.8539 | 0.2076 | 0.9418 | RTX 4090 | `results/slurm/drkg_full_bsage_mean_3236475_0.json` |
| 1 | `baseline_sage` | max (pool) | 0.8766 | 0.8766 | 0.2424 | 0.9555 | V100 | `results/slurm/drkg_full_bsage_pool_3236475_1.json` |
| 2 | `edge_aware_sage` (FiLM) | mean | — | — | — | — | (pending) | *(awaiting JSON)* |
| 3 | `edge_aware_sage` (FiLM) | max (pool) | — | — | — | — | (pending) | *(awaiting JSON)* |

**Slurm job IDs:** baselines `3236475_0`, `3236475_1`; edge-aware FiLM tasks `3236475_2`, `3236475_3` (metrics pending until JSON is written).

---

## Key Discoveries

### 1. OpenAI Embeddings Don't Capture Semantic Opposition

Initial analysis showed that OpenAI embeddings don't distinguish between opposing relation types:

| Relation Pair | Cosine Similarity | Expected |
|---------------|-------------------|----------|
| ACTIVATOR ↔ INHIBITOR | 0.74 | ~0 |
| AGONIST ↔ ANTAGONIST | 0.91 | ~0 |
| PARTIAL AGONIST ↔ BLOCKER | 0.74 | ~0 |

**Problem**: OpenAI embeddings treat opposing relations as very similar (near 0.8), not opposite.

### 2. SapBERT Shows Better Opposition Separation

SapBERT (biomedical domain-specific) shows much better separation:

| Relation Pair | SapBERT Cosine |
|---------------|----------------|
| ACTIVATOR ↔ INHIBITOR | 0.04 |
| AGONIST ↔ ANTAGONIST | 0.36 |
| PARTIAL AGONIST ↔ BLOCKER | -0.09 (negative!) |

### 3. Canonical Text Mode Hurts SapBERT

We implemented a "canonical" text mode that rewrites glossary entries into clean biomedical sentences:

- **raw**: "Interaction type: antagonism. Description: An antagonist..."
- **canonical**: "A compound blocks or inhibits a gene product."

However, canonical mode hurt SapBERT significantly:

| Config | Avg Cosine (Opposing Pairs) |
|--------|----------------------------|
| raw+sapbert | 0.07 |
| canonical+sapbert | 0.72 |
| raw+openai | 0.86 |
| canonical+openai | 0.75 |

**Conclusion**: The raw glossary text with all its "noise" (source tags, formatting) actually helps SapBERT distinguish relations better than cleaned canonical text.

### 4. Relation Diversity Matters

DRKG has a critical ordering issue - the first 25k rows contain only 1 relation type (bioarx::HumGenHumGen:Gene:Gene). We created a shuffled version to get relation-diverse slices.

**Low diversity (1 relation)**:
- baseline_sage: ROC 0.6954
- edge_aware_concat_random: ROC 0.6791
- edge_aware_concat_semantic: ROC 0.6540

**High diversity (99 relations)**:
- baseline_sage: ROC 0.5454
- edge_aware_concat_random: ROC 0.6079
- edge_aware_concat_semantic: ROC 0.6036

**Insight**: Edge-aware methods beat baseline by +10-11% on high-diversity data, but underperform on low-diversity data.

### 5. Semantic vs Random Ablation

On high-diversity shuffled data:

| Method | ROC-AUC | H@10 |
|--------|---------|------|
| edge_aware (semantic frozen) | 0.6036 | 0.6051 |
| edge_aware (shuffled semantic) | 0.6036 | 0.6030 |
| edge_aware (random frozen) | 0.6079 | 0.6053 |

**Insight**: semantic ≈ shuffled-semantic ≈ random. The learned projection dominates over frozen relation embeddings.

---

## Full Cosine Similarity Analysis

### Opposing Relation Pairs by Cache

| Pair | raw+openai | canonical+openai | raw+sapbert | canonical+sapbert |
|------|------------|-----------------|-------------|-------------------|
| AGONIST ↔ ANTAGONIST | 0.91 | 0.69 | 0.36 | 0.67 |
| ACTIVATOR ↔ INHIBITOR | 0.74 | 0.81 | **0.04** | 0.74 |
| ACTIVATOR ↔ BLOCKER | 0.73 | 0.68 | **-0.03** | 0.70 |
| GNBR A+ ↔ A- | 0.91 | 0.74 | 0.24 | 0.77 |
| GNBR E+ ↔ E- | 0.93 | 0.80 | **-0.14** | 0.82 |
| Hetionet CuG ↔ CdG | 0.92 | 0.84 | **-0.29** | 0.77 |
| Hetionet AuG ↔ AdG | 0.91 | 0.74 | 0.07 | 0.61 |
| Hetionet DuG ↔ DdG | 0.93 | 0.85 | 0.16 | 0.83 |
| STRING ACTIVATION ↔ INHIBITION | 0.79 | 0.81 | 0.21 | 0.69 |
| GNBR Pa ↔ J | 0.86 | 0.55 | 0.06 | 0.60 |
| **AVERAGE** | 0.86 | 0.75 | **0.07** | 0.72 |

**Winner**: raw+sapbert (average 0.07 - near perfect separation!)

---

## Experiment Results Summary

### Low Diversity (25k prefix - 1 relation)

| Method | ROC-AUC | H@10 |
|--------|---------|------|
| baseline_sage | **0.6954** | **0.7623** |
| edge_aware_concat_random | 0.6791 | 0.7300 |
| edge_aware_concat_semantic | 0.6540 | 0.6879 |

### High Diversity (25k shuffled - 99 relations)

| Method | ROC-AUC | H@10 | Delta vs Baseline |
|--------|---------|------|-------------------|
| baseline_sage | 0.5404 | 0.5367 | — |
| edge_aware_concat_random | 0.6085 | 0.6047 | +6.8% |
| edge_aware_concat_sapbert | 0.6078 | 0.6051 | +6.7% |
| edge_aware_concat_openai | 0.6036 | 0.6051 | +6.3% |
| edge_aware_gated_random | 0.6045 | 0.5988 | +6.4% |
| edge_aware_basis2_random | 0.6044 | 0.6066 | +6.4% |
| edge_aware_basis_no_align | 0.6046 | 0.6066 | +6.4% |
| edge_aware_film_sapbert | 0.6039 | 0.6093 | +6.4% |
| edge_aware_film_random | 0.6013 | 0.5954 | +6.1% |

### 10k Edge Comparisons (for quick experiments)

| Method | ROC-AUC | H@10 |
|--------|---------|------|
| baseline_sage (10k) | 0.5404 | 0.5367 |
| concat_random (10k) | 0.6085 | 0.6047 |
| basis_no_align (10k) | 0.6281 | 0.5543 |
| **basis_align_sapbert (10k)** | **0.6402** | **0.6039** |
| basis_align_random (10k) | 0.6287 | 0.5501 |
| film_sapbert (10ep) | 0.6464 | 0.6007 |
| film_random (10ep) | 0.6465 | 0.6192 |
| film_sapbert (50ep) | 0.6380 | 0.6055 |
| film_random (50ep) | 0.6461 | 0.6176 |

---

## Semantic Alignment Regularizer

Implemented for basis_mixture mode to preserve semantic geometry from frozen embeddings.

**Loss**: `L_align = Σ sim(semantic_i, semantic_j) * ||alpha_i - alpha_j||^2`

### Results (10k edges)

| Method | ROC-AUC | Delta | H@10 |
|--------|---------|-------|------|
| basis_no_align (10k) | 0.6281 | — | 0.5543 |
| basis_align_random (10k) | 0.6287 | +0.06% | 0.5501 |
| **basis_align_sapbert (10k)** | **0.6402** | **+1.2%** | **0.6039** |

**Key Finding**: The semantic alignment regularizer **only helps with semantic embeddings**:
- Random + alignment: essentially no improvement (+0.06%)
- SapBERT + alignment: +1.2% ROC, +5.0% H@10

This confirms the regularizer works as intended - it preserves semantic geometry from frozen embeddings. With random vectors, there's no semantic structure to preserve.

---

## FiLM (Feature-wise Linear Modulation)

Implemented FiLM edge relation mode that modulates learned base messages with gamma/beta scaling and shifting from relation embeddings.

**Formula**:
```
gamma = 1.0 + 0.1 * tanh(gamma_raw)  # Centered at identity
msg = gamma * msg_linear(x_j) + beta
```

### FiLM Results (10k edges)

| Method | ROC-AUC | H@10 |
|--------|---------|------|
| film_sapbert (10ep) | 0.6464 | 0.6007 |
| film_random (10ep) | 0.6465 | 0.6192 |
| film_sapbert (50ep) | 0.6380 | 0.6055 |
| film_random (50ep) | 0.6461 | 0.6176 |

### FiLM Results (25k edges)

| Method | ROC-AUC | Delta vs Baseline | H@10 |
|--------|---------|-------------------|------|
| baseline_sage (25k) | 0.5404 | — | 0.5367 |
| concat_random (25k) | 0.6085 | +6.8% | 0.6047 |
| concat_sapbert (25k) | 0.6078 | +6.7% | 0.6051 |
| film_sapbert (25k) | 0.6039 | +6.4% | 0.6093 |
| film_random (25k) | 0.6013 | +6.1% | 0.5954 |

### Key FiLM Findings

1. **FiLM performs comparably to concat**: Both film_sapbert (0.6039) and film_random (0.6013) are within 1% of concat_random (0.6085)

2. **No significant semantic vs random difference**: film_sapbert ≈ film_random, similar to other edge-aware methods

3. **FiLM slightly worse than concat**: FiLM underperforms concat by ~0.5% ROC-AUC

4. **All edge-aware methods beat baseline by ~6%**: The learned projection dominates regardless of relation mechanism

---

## Caches Generated

| Cache File | Text Mode | Embedding Model | Dim | Relations |
|------------|-----------|-----------------|-----|-----------|
| drkg-shuffled-seed42-relations-25k-semantic.pt | raw | OpenAI | 1536 | 99 |
| drkg-shuffled-25k-canonical.pt | canonical | OpenAI | 1536 | 99 |
| drkg-shuffled-25k-sapbert.pt | raw | SapBERT | 768 | 99 |
| drkg-shuffled-25k-canonical-sapbert.pt | canonical | SapBERT | 768 | 99 |

---

## Files Modified

- `src/kgml_new/embeddings/semantic.py` - Added canonical text mode, SapBERT support, similarity matrix computation
- `src/kgml_new/scripts/generate_relation_embeddings.py` - Added `--embedding-model` and `--relation-text-mode` flags
- `src/kgml_new/scripts/run_gpu_method.py` - Added `--embedding-model`, `--sapbert-model`, `--semantic-alignment-lambda`, `--glossary-path` flags
- `src/kgml_new/models/edge_aware_sage.py` - Added basis_mixture semantic alignment support, FiLM implementation
- `src/kgml_new/training/link_unsupervised.py` - Added semantic alignment regularizer to training loop

---

## Conclusions

1. **Semantic embeddings don't help** - The learned projection layer dominates regardless of frozen relation embedding initialization

2. **SapBERT is better at semantic separation** - But this advantage doesn't translate to better final performance

3. **All edge-aware mechanisms perform similarly** - concat, gated, basis_mixture, and FiLM all achieve ~6% improvement over baseline

4. **Semantic alignment regularizer helps** - But only with semantic embeddings and only for basis_mixture

5. **FiLM is a viable alternative** - Performs comparably to concat but doesn't improve on it

6. **The core bottleneck is the learned projection** - The projection from full relation embedding to edge_dim can learn arbitrary transformations that override any semantic structure

### What Would Actually Help

1. **Freeze the projection layer** - Test if frozen semantic embeddings help when projection can't override them
2. **Use richer edge features** - Current model uses only relation embedding, could add node type information
3. **Pre-train relation embeddings on opposition detection** - Fine-tune SapBERT on AGONIST/ANTAGONIST pairs specifically
4. **Different decoder architecture** - Current dot-product decoder may not leverage relation information well
