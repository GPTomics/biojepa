# Evaluation Planning

This document outlines additional evaluations beyond our current benchmark (predicting gene expression for held-out perturbations in K562). Each evaluation should have biological relevance, tell a coherent story, and where possible connect to our defined use cases.

Evaluations are ordered from easiest to most complex to implement.

## Current Evaluation

**Perturbation Response Prediction**: Given a control cell and a held-out perturbation, predict the resulting gene expression. Metrics: Global MSE, Pearson R (Top 20 DEGs), R² (all genes), R² (Top 50 DEGs).

---

## Proposed Evaluations

### 1. Direction of Effect Prediction
**Complexity**: Trivial (same data, different metric)

**Biological question**: Even when magnitude is wrong, does the model get the direction right?

**Setup**:
- For held-out perturbations, compute ground-truth direction (up/down/unchanged) for each gene
- Compute predicted direction from model output
- Measure: Accuracy, F1 for up/down/unchanged classes, focusing on top DEGs

**Story**: Getting direction right is often more actionable than exact magnitude. A drug developer cares whether a toxicity gene goes up or down. This evaluation is more forgiving of noise while still testing causal understanding.

**Data requirements**: Same as current eval.

---

### 2. Top DEG Recovery
**Complexity**: Trivial (same data, rank comparison)

**Biological question**: Does the model identify which genes are most affected by a perturbation?

**Setup**:
- For each held-out perturbation, rank genes by predicted absolute expression change
- Compare to ground-truth ranking of DEGs
- Measure: Precision@K, NDCG, overlap of top-50 predicted vs. top-50 actual DEGs

**Story**: Biologists care most about the genes that change dramatically. Even if the model's exact values are off, recovering the right set of top DEGs is practically useful for hypothesis generation.

**Data requirements**: Same as current eval.

---

### 3. Perturbation Retrieval
**Complexity**: Easy (same data, similarity ranking) | **Use Case 1: Target Discovery**

**Biological question**: Given a desired cellular outcome, can we identify which perturbation would achieve it?

**Setup**:
- Take held-out perturbed cells and compute their ground-truth expression delta from control
- Query the model: for each perturbation in our bank, predict the expression delta
- Rank perturbations by similarity to the ground-truth delta
- Measure: Recall@K (is the true perturbation in top K?), Mean Reciprocal Rank

**Story**: This directly tests the target discovery use case. If a user has a disease signature and wants to find what perturbation reverses it, the model needs to correctly rank perturbations by their predicted effect similarity.

**Data requirements**: Same as current eval, just different metric computation.

---

### 4. Uncertainty Calibration
**Complexity**: Easy (model already outputs variance) | **Use Case 9: Experiment Prioritization**

**Biological question**: Are the model's confidence estimates meaningful?

**Setup**:
- Model outputs mean and variance for predictions
- Bin predictions by predicted variance (low/medium/high uncertainty)
- Measure actual error in each bin
- Metrics: Expected Calibration Error, reliability diagrams
- Also: Do novel/rare perturbations have higher predicted uncertainty?

**Story**: For active learning and experiment prioritization, we need to trust that high uncertainty means "I don't know" not "random noise." Well-calibrated uncertainty enables rational experiment selection.

**Data requirements**: Same as current eval, plus analysis code.

---

### 5. Batch Effect Invariance
**Complexity**: Easy (have batch labels, simple classifier)

**Biological question**: Are the learned representations confounded by technical artifacts?

**Setup**:
- Train a classifier on frozen cell embeddings to predict batch ID
- Measure: Accuracy (lower is better—means batch is not encoded)
- Compare to a classifier predicting perturbation ID (should be high)

**Story**: We want representations that capture biology, not technical variation. If batch is easily predictable, the model may be learning shortcuts. If perturbation is predictable but batch isn't, we're capturing the right signal.

**Data requirements**: Batch labels (already in data).

---

### 6. Perturbation Severity Prediction
**Complexity**: Easy (compute from existing predictions)

**Biological question**: Can the model predict how "severe" a perturbation is?

**Setup**:
- Define severity as magnitude of expression change (L2 norm of delta)
- Compare predicted severity (L2 of predicted delta) to actual severity
- Optional: correlate with DepMap fitness scores if available
- Measure: Pearson/Spearman correlation of predicted vs actual severity

**Story**: Some perturbations barely affect the cell; others are catastrophic. If the model captures this, we can quickly screen perturbations for expected impact magnitude before analyzing full expression profiles.

**Data requirements**: Expression deltas (have). Optional: DepMap fitness scores.

---

### 7. Gene Embedding Pathway Recovery
**Complexity**: Moderate (need pathway annotations, clustering analysis)

**Biological question**: Do the learned gene embeddings capture known biological relationships?

**Setup**:
- Extract learned gene embeddings from CellStateEncoder
- For genes in known pathways (KEGG, Reactome), measure clustering quality
- Metrics: Silhouette score by pathway, Adjusted Rand Index vs. pathway labels
- Alternative: k-NN accuracy (do nearest neighbors share pathway membership?)

**Story**: The encoder should learn that genes which function together are represented similarly. This is a sanity check that the latent space has biological structure, not just statistical regularities.

**Data requirements**: Pathway membership annotations (KEGG, Reactome, GO).

---

### 8. Action Vector Interpretability
**Complexity**: Moderate (need pathway annotations for interpretation)

**Biological question**: Does the ActionComposer learn meaningful perturbation representations?

**Setup**:
- Extract action vectors for all perturbations
- Cluster action vectors and analyze cluster composition
- Measure: Do perturbations targeting same pathway cluster together? Do perturbations with similar phenotypic effects cluster?
- Visualize with UMAP, annotate by pathway/function

**Story**: The action space is where perturbation "meaning" lives. If it's interpretable, we can potentially do arithmetic (drug A + drug B) or identify novel perturbations that would fill gaps in the action space.

**Data requirements**: Pathway annotations for target genes.

---

### 9. Essential Gene Prediction
**Complexity**: Moderate (need DepMap download, linear probe)

**Biological question**: Do the learned representations encode functional importance?

**Setup**:
- Get gene essentiality scores from DepMap (CRISPR dependency scores for K562)
- Train a linear probe on frozen gene embeddings to predict essentiality
- Measure: Pearson correlation, AUROC for essential vs. non-essential classification

**Story**: If the model learns which genes are critical for cell viability, it suggests understanding of the functional hierarchy. This is prerequisite for synthetic lethality prediction.

**Data requirements**: DepMap CRISPR scores for K562 (publicly available).

---

### 10. Mechanism of Action Matching
**Complexity**: Moderate (need pathway annotations, similarity analysis) | **Use Case 4: MoA Inference**

**Biological question**: Do perturbations with similar mechanisms produce similar predicted effects?

**Setup**:
- Group perturbations by their known pathway membership (e.g., all genes in "Cell Cycle", "DNA Repair", "Apoptosis")
- For each perturbation, compute the predicted expression delta
- Measure within-pathway vs. between-pathway similarity of predicted deltas
- Alternative: Use known drug-target relationships from ChEMBL/DrugBank if chemical perturbation data is added

**Story**: If the model learns causal biology, perturbations hitting the same pathway should produce correlated expression changes. This validates that action vectors encode mechanistic information, not just identity.

**Data requirements**: Pathway annotations for target genes (KEGG, Reactome, GO).

---

### 11. Cross-Cell-Type Transfer
**Complexity**: Complex (need to process new dataset, full pipeline)

**Biological question**: Does the model learn universal cell physics or K562-specific patterns?

**Setup**:
- Train on K562, evaluate zero-shot on RPE1 (or another cell line)
- Same metrics as current eval
- Compare: K562→K562 vs. K562→RPE1 performance drop

**Story**: If the model truly learns causal biology, some of that should transfer. Complete failure suggests overfitting to K562 idiosyncrasies. Partial transfer suggests shared biology is captured.

**Data requirements**: RPE1 perturbation data (Replogle RPE1 Essential).

---

### 12. Synthetic Lethality Signal
**Complexity**: Most Complex (architectural limitations, speculative) | **Use Case 5: Synthetic Lethality**

**Biological question**: Can the model identify known synthetic lethal pairs?

**Setup**:
- Get known synthetic lethal pairs from literature/databases (e.g., SynLethDB)
- For each pair (A, B), predict effect of A alone, B alone, and approximate A+B
- Measure: Do known SL pairs show predicted non-additive lethality?
- This is exploratory—current model doesn't handle combinations, but we can look for signal

**Story**: This tests whether the model's learned physics captures the non-linear interactions that underlie synthetic lethality. Even rough signal here would be exciting and validate the drug combination use case.

**Data requirements**: Synthetic lethality database, method to approximate combination effects (model architecture limitation).

---

## Implementation Phases

### Phase 1: Quick Wins (1-6)
Use existing data and model outputs. Minimal new code. Can implement in a single notebook.
- Direction of Effect, Top DEG Recovery, Perturbation Retrieval
- Uncertainty Calibration, Batch Effect Invariance, Perturbation Severity

### Phase 2: Biological Annotations (7-10)
Require downloading and mapping pathway annotations (KEGG/Reactome/GO) and DepMap scores.
- Gene Embedding Pathway Recovery, Action Vector Interpretability
- Essential Gene Prediction, Mechanism of Action Matching

### Phase 3: New Data (11)
Requires processing additional perturbation datasets.
- Cross-Cell-Type Transfer (RPE1)

### Phase 4: Architecture Extensions (12)
Requires model changes to handle perturbation combinations.
- Synthetic Lethality Signal

---

## Data Dependencies Summary

| Data | Source | Evaluations |
|------|--------|-------------|
| Current eval data | Already have | 1-6 |
| KEGG/Reactome/GO pathways | Public APIs | 7, 8, 10 |
| DepMap K562 CRISPR | DepMap Portal | 9, (6 optional) |
| Replogle RPE1 | GEARS/GEO | 11 |
| SynLethDB | Public database | 12 |
