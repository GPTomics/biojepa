# Evaluation Planning

This document outlines evaluations for BioJEPA. Each evaluation has biological relevance, tells a coherent story, and where possible connects to our defined use cases.

All evaluations are implemented in both `v0_4/evals/` and `v0_5/evals/` with identical structure. The only difference is which model checkpoint and imports each version uses.

---

## Implemented Evaluations

### eval_1: Expression Prediction + Severity
**Notebook**: `eval_1_expression_prediction.ipynb`

**Biological question**: Can we predict the gene expression profile after a perturbation? Can we predict how severe the perturbation's effect will be?

**Metrics**:

| Metric | Level | Description |
|--------|-------|-------------|
| Global MSE | Sample | Mean squared error between predicted and true expression deltas |
| Pearson R (Top 20 DEGs) | Sample | Correlation on the 20 genes with largest true changes |
| R² (All Genes) | Perturbation | Coefficient of determination across all genes (averaged per perturbation) |
| R² (Top 50 DEGs) | Perturbation | R² on the 50 genes with largest true changes |
| Severity Pearson | Perturbation | Correlation between predicted and true L2 norm of delta |
| Severity Spearman | Perturbation | Rank correlation of severity |
| MAE by Magnitude | Gene | Mean absolute error binned by true change magnitude |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Global MSE | < 0.3 | 0.3 - 0.7 | > 0.7 | Lower is better. Depends on data normalization |
| Pearson R (Top 20) | > 0.8 | 0.5 - 0.8 | < 0.5 | Measures if top DEGs move in right direction with right relative magnitude |
| R² (All Genes) | > 0.85 | 0.7 - 0.85 | < 0.7 | High values expected since most genes don't change much |
| R² (Top 50 DEGs) | > 0.3 | 0.0 - 0.3 | < 0.0 | This is the hard test - predicting genes that actually change. Negative R² means worse than predicting the mean |
| Severity Pearson | > 0.7 | 0.4 - 0.7 | < 0.4 | Can we tell big effects from small effects? |
| Severity Spearman | > 0.5 | 0.3 - 0.5 | < 0.3 | Rank ordering of perturbation severity |

**Interpretation guide**:
- R² on all genes will always look good because ~95% of genes barely change - the model just needs to predict "no change" for most genes
- R² on Top 50 DEGs is the real test - these are the genes that matter biologically
- Negative R² on DEGs means the model is actively wrong about the genes that change most
- Severity correlation tells you if the model knows which perturbations are "big deals" vs subtle
- MAE by magnitude shows where the model struggles - typically errors scale with true change magnitude

---

### eval_2: Gene-Level Analysis (Direction + DEG Recovery)
**Notebook**: `eval_2_gene_level_analysis.ipynb`

**Biological question**: Even when magnitude is wrong, does the model get the direction right? Does it identify which genes are most affected?

**Part A - Direction of Effect**:

| Metric | Description |
|--------|-------------|
| Direction Accuracy (All) | Fraction of genes with correct UP/DOWN/UNCHANGED classification |
| Direction Accuracy (Top 50) | Accuracy on the 50 most-changed genes |
| Direction F1 (per class) | F1 score for UP, DOWN, UNCHANGED classes |
| Accuracy by Magnitude | Direction accuracy binned by true change magnitude |

**Part B - Top DEG Recovery**:

| Metric | Description |
|--------|-------------|
| Precision@K | Fraction of predicted top-K that are truly top-K |
| NDCG@K | Normalized Discounted Cumulative Gain (ranking quality) |
| Overlap | Number of genes in both predicted and true top-K |
| vs Random | Improvement over random baseline |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Direction Accuracy (All) | > 0.85 | 0.7 - 0.85 | < 0.7 | High baseline from UNCHANGED class |
| Direction Accuracy (Top 50) | > 0.7 | 0.5 - 0.7 | < 0.5 | Chance is ~0.33 for 3 classes |
| Direction F1 (UP class) | > 0.6 | 0.4 - 0.6 | < 0.4 | Detecting upregulation |
| Direction F1 (DOWN class) | > 0.6 | 0.4 - 0.6 | < 0.4 | Detecting downregulation |
| Precision@20 | > 0.4 | 0.2 - 0.4 | < 0.2 | Are predicted top DEGs actually top DEGs? |
| Precision@50 | > 0.3 | 0.15 - 0.3 | < 0.15 | Harder at larger K |
| NDCG@K | > 0.6 | 0.4 - 0.6 | < 0.4 | 1.0 = perfect ranking, 0 = random |
| vs Random | > 5x | 2-5x | < 2x | Improvement over chance |

**Interpretation guide**:
- Direction is often more actionable than magnitude - a drug developer cares if a toxicity gene goes UP or DOWN
- High direction accuracy with low magnitude accuracy suggests the model understands causal structure but struggles with quantitative precision
- Precision@K directly measures "if I look at the model's top predictions, how many are real hits?"
- Low Precision@K but high NDCG means the model ranks well overall but the very top predictions are noisy
- Accuracy by magnitude typically shows a U-shape: good on unchanged genes (easy), poor on moderate changes (ambiguous), good again on large changes (clear signal)

---

### eval_3: Perturbation Retrieval
**Notebook**: `eval_3_perturbation_retrieval.ipynb`

**Biological question**: Given a desired cellular outcome, can we identify which perturbation would achieve it?

**Use Case**: Target Discovery - finding perturbations that reverse disease signatures

**Metrics**:

| Metric | Description |
|--------|-------------|
| Recall@K | Is the true perturbation in the top K predictions? |
| Mean Reciprocal Rank (MRR) | Average of 1/rank across all queries |
| Median Rank | Typical rank of true perturbation |
| Mean Rank | Average rank of true perturbation |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Recall@1 | > 0.3 | 0.1 - 0.3 | < 0.1 | Exact match at rank 1 |
| Recall@5 | > 0.6 | 0.3 - 0.6 | < 0.3 | True perturbation in top 5 |
| Recall@10 | > 0.75 | 0.5 - 0.75 | < 0.5 | True perturbation in top 10 |
| Recall@50 | > 0.9 | 0.7 - 0.9 | < 0.7 | True perturbation in top 50 |
| MRR | > 0.5 | 0.25 - 0.5 | < 0.25 | Higher = better. 1.0 = always rank 1 |
| Median Rank | < 10 | 10 - 100 | > 100 | Lower is better |

**Interpretation guide**:
- This eval simulates the target discovery workflow: "I have a phenotype, what causes it?"
- Recall@K answers "if I test the top K predictions, will I find the right answer?"
- With ~1250 perturbations, random Recall@10 ≈ 0.008, so even modest performance is meaningful
- MRR penalizes late ranks heavily - an MRR of 0.5 means the true perturbation is typically ranked around position 2
- High Recall@10 but low Recall@1 suggests the model narrows down candidates but can't pinpoint exactly
- Poor retrieval often indicates the model predicts similar deltas for many perturbations (low specificity)

---

### eval_4: Uncertainty Calibration
**Notebook**: `eval_4_uncertainty_calibration.ipynb`

**Biological question**: Are the model's confidence estimates meaningful?

**Use Case**: Experiment Prioritization - trusting uncertainty for active learning

**Metrics**:

| Metric | Level | Description |
|--------|-------|-------------|
| Uncertainty-Error Pearson | Sample | Correlation between predicted uncertainty and actual error |
| Uncertainty-Error Spearman | Sample | Rank correlation (more robust to outliers) |
| Expected Calibration Error (ECE) | Sample | Average gap between confidence and accuracy |
| Monotonicity Score | Sample | % of bins where error increases with uncertainty |
| Pert-Level Pearson | Perturbation | Correlation at perturbation level |
| Pert-Level Spearman | Perturbation | Rank correlation at perturbation level |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Uncertainty-Error Pearson | > 0.3 | 0.1 - 0.3 | < 0.1 | Positive = uncertainty predicts error |
| Uncertainty-Error Spearman | > 0.3 | 0.1 - 0.3 | < 0.1 | Rank correlation, more robust |
| ECE | < 0.1 | 0.1 - 0.2 | > 0.2 | Lower is better. 0 = perfectly calibrated |
| Monotonicity Score | > 80% | 60-80% | < 60% | % of bins where error increases with uncertainty |

**Interpretation guide**:
- A well-calibrated model says "I'm uncertain" when it's actually wrong more often
- Positive correlation means uncertainty is informative - you can trust it for experiment prioritization
- ECE measures the gap between predicted confidence and actual accuracy across bins
- Monotonicity checks if error consistently increases as uncertainty increases (it should)
- Zero or negative correlation means uncertainty is meaningless noise - don't use it for decisions
- Note: Uncertainty is in latent space (mean of z_pred_logvar), not gene space, so correlations may be modest even if useful

---

### eval_5: Batch Effect Invariance
**Notebook**: `eval_5_batch_invariance.ipynb`

**Biological question**: Are the learned representations confounded by technical artifacts?

**Metrics**:

| Metric | Description |
|--------|-------------|
| Batch Classifier Accuracy | Linear probe accuracy predicting batch ID from embeddings |
| Batch Above Chance Ratio | Batch accuracy / chance (1/n_batches) |
| Perturbation Classifier Accuracy | Linear probe accuracy predicting perturbation ID |
| Perturbation Above Chance Ratio | Pert accuracy / chance (1/n_perts) |
| Invariance Ratio | Pert accuracy / Batch accuracy |

**How to interpret**:

| Metric | Good | Average | Concerning | Notes |
|--------|------|---------|------------|-------|
| Batch Above Chance | < 2x | 2-5x | > 5x | Want batch to be unpredictable |
| Pert Above Chance | > 10x | 5-10x | < 5x | Want perturbation to be predictable |
| Invariance Ratio | > 5 | 2 - 5 | < 2 | Pert accuracy / Batch accuracy |

**Interpretation guide**:
- Chance level depends on number of classes (1/N for N classes)
- Batch accuracy at chance means embeddings contain zero batch information - ideal
- Batch accuracy >> chance means technical artifacts leak into representations - concerning for generalization
- Perturbation accuracy should be high - this confirms embeddings encode biological signal
- The ratio tells you how much more "biological" than "technical" your representations are
- Example: 48 batches (chance=2.1%), 286 perts (chance=0.35%). Batch acc=2.9%, Pert acc=0.8% → Batch is 1.4x chance, Pert is 2.2x chance
- Note: Low perturbation accuracy isn't necessarily bad if embeddings capture perturbation effects rather than identity

---

### eval_6: Embedding Pathway Analysis
**Notebook**: `eval_6_embedding_pathways.ipynb`

**Biological question**: Do learned embeddings capture known biological relationships?

**Part A - Gene Embedding Pathway Recovery**: Do genes in the same pathway cluster together in the encoder's learned gene embeddings?

**Part B - Action Vector Interpretability**: Do perturbations targeting genes in the same pathway produce similar action vectors?

**Metrics**:

| Metric | Part | Description |
|--------|------|-------------|
| Silhouette Score (KEGG) | A, B | Clustering quality (-1 to 1) for KEGG pathways |
| Silhouette Score (Reactome) | A | Clustering quality for Reactome pathways |
| k-NN Accuracy (KEGG) | A, B | Fraction of k nearest neighbors from same pathway |
| k-NN Accuracy (Reactome) | A | Same, for Reactome |
| n_classes | A, B | Number of pathways with sufficient samples |
| n_samples | A, B | Number of genes/perturbations evaluated |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Silhouette Score | > 0.2 | 0.0 - 0.2 | < 0.0 | Higher = tighter pathway clusters. Negative = wrong clusters |
| k-NN Accuracy | > 3x chance | 1.5-3x chance | < 1.5x chance | Chance = 1/n_pathways |

**Interpretation guide**:
- High silhouette means genes/perturbations in same pathway are close together and far from other pathways
- Negative silhouette means genes are closer to other pathways than their own - embeddings don't capture pathway structure
- k-NN accuracy measures if nearest neighbors share pathway membership
- Gene embeddings (Part A) reflect encoder's learned gene relationships
- Action vectors (Part B) reflect composer's learned perturbation relationships
- Both should capture biological structure if the model learns meaningful representations
- Pathways are imperfect labels - genes belong to multiple pathways, so moderate scores are expected

**Data requirements**: Pathway annotations via gseapy (KEGG_2021_Human, Reactome_Pathways_2024).

---

### eval_7: Mechanism of Action Matching
**Notebook**: `eval_7_moa_matching.ipynb`

**Biological question**: Do perturbations targeting genes in the same pathway produce similar predicted expression changes?

**Use Case**: MoA Inference - match unknown perturbations to known mechanisms

**Metrics**:

| Metric | Description |
|--------|-------------|
| Mean Within-Pathway Similarity | Average cosine similarity of predicted deltas for same-pathway perturbations |
| Mean Between-Pathway Similarity | Average cosine similarity for different-pathway perturbations |
| Similarity Ratio | Within / Between |
| Mann-Whitney p-value | Statistical significance of difference |
| n_pathways | Number of pathways with >= 3 perturbations |
| n_perturbations | Total perturbations in analysis |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Similarity Ratio | > 1.5 | 1.1 - 1.5 | < 1.1 | Higher = same-pathway perts more similar |
| p-value | < 0.01 | 0.01 - 0.05 | > 0.05 | Statistical significance of difference |

**Interpretation guide**:
- If ratio > 1, perturbations hitting same pathway produce more similar predicted effects
- This validates that the model captures mechanistic relationships, not just perturbation identity
- High ratio means we could potentially use the model to infer MoA of unknown compounds
- p-value confirms the difference isn't due to chance
- Ratio ~1.0 means the model doesn't distinguish pathway-related perturbations from unrelated ones
- Note: This tests predictions, not embeddings - it's about whether the model's outputs respect biological structure

**Data requirements**: Pathway annotations via gseapy (KEGG_2021_Human).

---

### eval_8: Essential Gene Prediction
**Notebook**: `eval_8_essential_genes.ipynb`

**Biological question**: Do the learned gene embeddings encode functional importance?

**Setup**: Train a linear probe on frozen gene embeddings to predict gene essentiality scores from DepMap (CRISPR dependency scores for K562).

**Metrics**:

| Metric | Split | Description |
|--------|-------|-------------|
| Pearson r | Train/Test | Correlation between predicted and true essentiality score |
| Spearman r | Train/Test | Rank correlation (robust to outliers) |
| AUROC | Train/Test | Classification performance (essential vs non-essential at -0.5 threshold) |
| n_essential | Test | Number of essential genes in test set |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Pearson r (test) | > 0.3 | 0.1 - 0.3 | < 0.1 | Correlation with true essentiality |
| Spearman r (test) | > 0.3 | 0.1 - 0.3 | < 0.1 | Rank correlation |
| AUROC (test) | > 0.7 | 0.55 - 0.7 | < 0.55 | 0.5 = random |

**Interpretation guide**:
- High correlation means gene embeddings encode which genes are critical for cell viability
- This is a prerequisite for synthetic lethality prediction
- Embeddings that predict essentiality likely capture functional hierarchy
- Low scores suggest embeddings encode expression patterns but not functional importance
- Note: Essentiality is K562-specific, so this tests cell-type-specific functional encoding
- AUROC > 0.7 with Pearson > 0.25 suggests embeddings have meaningful biological content

**Data requirements**: DepMap CRISPR scores auto-downloaded via DepMap API.

---

## Evaluations On Hold

These evaluations require additional data or architectural changes and are not currently prioritized.

### eval_9: Cross-Cell-Type Transfer
**Status**: On Hold - requires new dataset processing

**Biological question**: Does the model learn universal cell physics or K562-specific patterns?

**Setup**:
- Train on K562, evaluate zero-shot on RPE1 (or another cell line)
- Same metrics as eval_1 (expression prediction)
- Compare: K562→K562 vs. K562→RPE1 performance drop

**What's needed**:
- Download and process Replogle RPE1 Essential dataset from GEARS/GEO
- Run same preprocessing pipeline as K562
- Match gene sets between cell types

**Story**: If the model truly learns causal biology, some of that should transfer. Complete failure suggests overfitting to K562 idiosyncrasies. Partial transfer suggests shared biology is captured.

---

### eval_10: Synthetic Lethality Signal
**Status**: On Hold - requires architectural changes

**Biological question**: Can the model identify known synthetic lethal pairs?

**Setup**:
- Get known synthetic lethal pairs from literature/databases (e.g., SynLethDB)
- For each pair (A, B), predict effect of A alone, B alone, and approximate A+B
- Measure: Do known SL pairs show predicted non-additive lethality?
- This is exploratory - current model doesn't handle combinations

**What's needed**:
- Synthetic lethality database (SynLethDB)
- Method to approximate combination effects:
  - Option 1: Add action vectors (hacky but tests for signal)
  - Option 2: Modify ActionComposer to accept multiple perturbations (architectural change)

**Story**: This tests whether the model's learned physics captures the non-linear interactions that underlie synthetic lethality. Even rough signal here would be exciting and validate the drug combination use case.

---

## Implementation Summary

| Eval | Notebook | Biological Question | Key Metrics | Status |
|------|----------|---------------------|-------------|--------|
| 1 | eval_1_expression_prediction | Predict expression after perturbation | MSE, R², Severity correlation | Done |
| 2 | eval_2_gene_level_analysis | Direction + DEG identification | Direction accuracy, Precision@K, NDCG | Done |
| 3 | eval_3_perturbation_retrieval | Find perturbation from outcome | Recall@K, MRR | Done |
| 4 | eval_4_uncertainty_calibration | Are confidence estimates meaningful? | Uncertainty-Error correlation, ECE | Done |
| 5 | eval_5_batch_invariance | Batch vs biological signal | Classifier accuracies, Invariance ratio | Done |
| 6 | eval_6_embedding_pathways | Pathway structure in embeddings | Silhouette, k-NN accuracy | Done |
| 7 | eval_7_moa_matching | Same-pathway similarity | Within/between ratio, p-value | Done |
| 8 | eval_8_essential_genes | Functional importance in embeddings | Pearson r, AUROC | Done |
| 9 | - | Cross-cell-type transfer | Same as eval_1 | On Hold |
| 10 | - | Synthetic lethality detection | TBD | On Hold |

---

## Data Dependencies

| Data | Source | Evaluations | Status |
|------|--------|-------------|--------|
| K562 Essential test set | Already have | 1-8 | Done |
| Batch labels (gem_group) | Added to shards | 5 | Done |
| KEGG/Reactome pathways | gseapy | 6, 7 | Done |
| DepMap K562 CRISPR | DepMap API | 8 | Done |
| Replogle RPE1 | GEARS/GEO | 9 | Pending |
| SynLethDB | Public database | 10 | Pending |
