# BioJEPA v0.6 Code Audit

**Date:** 2026-01-28
**Scope:** Data prep, model training, evals, and eval_planning.md

---

## Bugs

### 1. Pretraining Evals Use Incorrect DataLoader API (evals/evals.py)

**Severity:** High - Will cause runtime errors

Several pretraining eval functions use the old v0.4/v0.5 dataloader API instead of the v0.6 namedtuple approach:

**Problem A:** Passing unsupported parameters

```python
# _batch_invariance (line 526) - return_batch_id doesn't exist in v0.6 TrainingLoader
test_loader = TrainingLoader(..., return_batch_id=True)

# _cell_type_probing (line 647) - return_cell_type doesn't exist
test_loader = TrainingLoader(..., return_cell_type=True)
```

**Problem B:** Incorrect tuple unpacking
```python
# _batch_invariance (line 532) - TrainingBatch has 14 fields, not 8
cont_x, cont_tot, case_x, case_tot, p_idx, p_mod, p_mode, batch_id = test_loader.next_batch()

# _reconstruction (line 699)
cont_x, cont_tot, _, _, _, _, _ = test_loader.next_batch()

# _perturbation_detection (line 763)
cont_x, cont_tot, case_x, case_tot, _, _, _ = test_loader.next_batch()

# _embedding_consistency (line 804)
cont_x, cont_tot, case_x, case_tot, p_idx, _, _ = test_loader.next_batch()

# _latent_space_health (line 857)
cont_x, cont_tot, _, _, _, _, _ = test_loader.next_batch()
```

**Correct pattern** (already used in `_run_test_inference`, line 388-391):
```python
batch = test_loader.next_batch()
cont_x, cont_tot = batch.control, batch.control_total
case_x, case_tot = batch.case, batch.case_total
# Access batch.batch_id, batch.cell_type directly - no special parameters needed
```

---

### 2. MODE_TO_ID / num_modes Mismatch (config_v0_6.py, biojepa_v0_6.py)

**Severity:** Medium - Will cause index out of bounds if `unknown` mode is used

**config_v0_6.py:**
```python
MODE_TO_ID = {
    'crispri': 0, 'crispra': 1, 'overexpression': 2, 'knockout': 3,
    'inhibitor': 4, 'agonist': 5, 'degrader': 6, 'binder': 7,
    'control': 8,   # <-- questionable
    'unknown': 9    # <-- out of bounds
}
```

**biojepa_v0_6.py (ActionComposerConfig):**
```python
num_modes: int = 9  # Valid indices: 0-8
```

If mode=9 (unknown) is ever used, `self.mode_embedding(mode_ids)` will fail with index out of bounds.

CLAUDE.md documents `unknown=8`, not `unknown=9`. The `control` mode is questionable since control samples don't go through the composer as perturbations.

**Resolution:** Remove `control` from MODE_TO_ID and set `unknown=8` to match CLAUDE.md

---

## Code Quality Issues

### 1. Duplicated Utility Functions !IGNORE !

`get_seq_embeddings()` and `get_target_embeddings()` appear in both:
- `training_v0_6.py` (lines 46-96)
- `evals/evals.py` (lines 36-61)

### 2. Misplaced Import (evals/evals.py:62)

```python
def get_seq_embeddings(...): ...
def get_target_embeddings(...): ...
from .pathway_utils import load_pathway_annotations, ...  # Import after functions
```

### 3. init_weights_robust Duplicated  !IGNORE !

Appears in both `biojepa_v0_6.py` and `evals/linear_expression_decoder.py`.

---

## eval_planning.md

### Minor Issue

**cell_type_probing status:** Doc says "Pending (v0.6)" in data dependencies but the eval code exists. The issue is that v0_4/v0_5 shards don't have cell_type data. Should clarify that the eval is implemented but requires v0.6 shards.

---

## Summary

| Category | Count |
|----------|-------|
| Bugs | 2 |
| Code Quality Issues | 3 |
| eval_planning.md Issues | 1 |

**Priority Fixes:**
1. Update pretraining evals to use namedtuple access pattern (high - currently broken)
2. Resolve MODE_TO_ID / num_modes inconsistency (medium - potential runtime error)
