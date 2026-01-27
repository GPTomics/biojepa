# Evaluation Planning

This document outlines evaluations for BioJEPA. Each evaluation has biological relevance, tells a coherent story, and where possible connects to our defined use cases.

---

## Evaluation Stages

**Pretraining Evals** (encoder-only, run after stage 1):
- `batch_invariance`: Are representations confounded by batch effects?
- `gene_embedding_pathways`: Do genes in same pathway cluster together?
- `essential_gene_prediction`: Do gene embeddings encode functional importance?

**Alignment Evals** (run after stage 2 - alignment):
- `alignment_recall`: Can we retrieve correct protein from DNA query?
- `modality_gap_analysis`: Do DNA/Protein cluster separately in action space?
- `anchor_input_consistency`: For same gene, do DNA/Protein produce similar actions?
- `mode_sensitivity`: Does FiLM conditioning on mode change embeddings?
- `target_family_probing`: Do action embeddings encode protein family?

**Full Model Evals** (run after stage 3 - full training):
- `expression_prediction`: Can we predict gene expression after perturbation?
- `gene_level_analysis`: Direction of effect + top DEG recovery
- `perturbation_retrieval`: Given desired outcome, find the perturbation
- `uncertainty_calibration`: Are confidence estimates meaningful?
- `action_vector_pathways`: Do same-pathway perturbations have similar action vectors?
- `moa_matching`: Do same-pathway perturbations produce similar predicted effects?

---

## Implemented Evaluations

### expression_prediction
**Notebook**: `expression_prediction.ipynb`

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

### gene_level_analysis
**Notebook**: `gene_level_analysis.ipynb`

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

### perturbation_retrieval
**Notebook**: `perturbation_retrieval.ipynb`

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
- With ~1250 perturbations, random Recall@10 = 0.008, so even modest performance is meaningful
- MRR penalizes late ranks heavily - an MRR of 0.5 means the true perturbation is typically ranked around position 2
- High Recall@10 but low Recall@1 suggests the model narrows down candidates but can't pinpoint exactly
- Poor retrieval often indicates the model predicts similar deltas for many perturbations (low specificity)

---

### uncertainty_calibration
**Notebook**: `uncertainty_calibration.ipynb`

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
| bin_mean_errors | Sample | Mean error in each uncertainty bin (for plotting calibration curves) |

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

### batch_invariance
**Notebook**: `batch_invariance.ipynb`

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
- Example: 48 batches (chance=2.1%), 286 perts (chance=0.35%). Batch acc=2.9%, Pert acc=0.8% -> Batch is 1.4x chance, Pert is 2.2x chance
- Note: Low perturbation accuracy isn't necessarily bad if embeddings capture perturbation effects rather than identity

---

### gene_embedding_pathways
**Notebook**: `gene_embedding_pathways.ipynb`

**Biological question**: Do genes in the same pathway cluster together in the encoder's learned gene embeddings?

**Metrics**:

| Metric | Description |
|--------|-------------|
| Silhouette Score (KEGG) | Clustering quality (-1 to 1) for KEGG pathways |
| Silhouette Score (Reactome) | Clustering quality for Reactome pathways |
| k-NN Accuracy (KEGG) | Fraction of k nearest neighbors from same pathway |
| k-NN Accuracy (Reactome) | Same, for Reactome |
| n_classes | Number of pathways with sufficient samples |
| n_samples | Number of genes evaluated |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Silhouette Score | > 0.2 | 0.0 - 0.2 | < 0.0 | Higher = tighter pathway clusters. Negative = wrong clusters |
| k-NN Accuracy | > 3x chance | 1.5-3x chance | < 1.5x chance | Chance = 1/n_pathways |

**Interpretation guide**:
- High silhouette means genes in same pathway are close together and far from other pathways
- Negative silhouette means genes are closer to other pathways than their own - embeddings don't capture pathway structure
- k-NN accuracy measures if nearest neighbors share pathway membership
- Gene embeddings reflect encoder's learned gene relationships
- Pathways are imperfect labels - genes belong to multiple pathways, so moderate scores are expected

**Data requirements**: Pathway annotations via gseapy (KEGG_2021_Human, Reactome_Pathways_2024).

---

### action_vector_pathways
**Notebook**: `action_vector_pathways.ipynb`

**Biological question**: Do perturbations targeting genes in the same pathway produce similar action vectors?

**Metrics**:

| Metric | Description |
|--------|-------------|
| Silhouette Score (KEGG) | Clustering quality (-1 to 1) for KEGG pathways |
| k-NN Accuracy (KEGG) | Fraction of k nearest neighbors from same pathway |
| n_classes | Number of pathways with sufficient samples |
| n_samples | Number of perturbations evaluated |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Silhouette Score | > 0.2 | 0.0 - 0.2 | < 0.0 | Higher = tighter pathway clusters. Negative = wrong clusters |
| k-NN Accuracy | > 3x chance | 1.5-3x chance | < 1.5x chance | Chance = 1/n_pathways |

**Interpretation guide**:
- Action vectors reflect composer's learned perturbation relationships
- High silhouette means perturbations hitting same pathway produce similar action vectors
- This tests whether the composer learns meaningful biological structure

**Data requirements**: Pathway annotations via gseapy (KEGG_2021_Human).

---

### moa_matching
**Notebook**: `moa_matching.ipynb`

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

### essential_gene_prediction
**Notebook**: `essential_gene_prediction.ipynb`

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

### cell_type_probing
**Implementation**: `evals/evals.py`

**Biological question**: Does the encoder disentangle biological cell state from noise?

**Setup**: Train logistic regression on cell embeddings (mean-pooled across genes) to predict cell type.

**Metrics**:

| Metric | Description |
|--------|-------------|
| Accuracy | Fraction of correctly classified cells |
| Macro F1 | F1 score averaged across cell types (handles class imbalance) |
| Above Chance Ratio | Accuracy / (1 / n_cell_types) |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Accuracy | > 0.9 | 0.7 - 0.9 | < 0.7 | Depends on n_cell_types |
| Macro F1 | > 0.8 | 0.5 - 0.8 | < 0.5 | More robust to class imbalance |
| Above Chance | > 5x | 2-5x | < 2x | Higher = better separation |

**Interpretation guide**:
- High accuracy means the encoder preserves cell-type-specific biology
- This is a prerequisite for cross-cell-type transfer learning
- Low accuracy with good expression prediction suggests the encoder focuses on perturbation effects over cell state
- Note: Requires multi-cell-type data. Returns early with error if only 1 cell type present.

**Data requirements**: `cell_type` field in training shards, `cell_type_to_id.json` mapping.

---

### reconstruction
**Implementation**: `evals/evals.py`

**Biological question**: Does the encoder preserve fine-grained expression information, or does VICReg wash it out?

**Setup**: Train 1-layer MLP to reconstruct gene expression from per-gene embeddings. Use 80% of genes for training, 20% for testing reconstruction capability.

**Metrics**:

| Metric | Description |
|--------|-------------|
| Reconstruction MSE | Mean squared error on held-out genes |
| Pearson R | Correlation between predicted and true expression |
| R² | Coefficient of determination |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Pearson R | > 0.8 | 0.5 - 0.8 | < 0.5 | Measures relative structure preservation |
| R² | > 0.6 | 0.3 - 0.6 | < 0.3 | Measures absolute reconstruction quality |

**Interpretation guide**:
- High reconstruction = encoder preserves expression information
- Low reconstruction with good downstream performance = encoder compresses to task-relevant features
- Very low reconstruction is a red flag - embeddings may be degenerate
- This is a sanity check for VICReg variance constraints not being too aggressive

**Data requirements**: Existing data only.

---

### perturbation_detection
**Implementation**: `evals/evals.py`

**Biological question**: Is the perturbation signal preserved in the latent space, or has VICReg washed it out as "noise"?

**Setup**: Train binary classifier to distinguish control cell embeddings from perturbed cell embeddings.

**Metrics**:

| Metric | Description |
|--------|-------------|
| AUROC | Area under ROC curve for binary classification |
| Accuracy | Fraction correctly classified |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| AUROC | > 0.7 | 0.55 - 0.7 | < 0.55 | 0.5 = random chance |
| Accuracy | > 0.65 | 0.55 - 0.65 | < 0.55 | Chance = 0.5 |

**Interpretation guide**:
- High AUROC = perturbation effects survive encoding into latent space
- Low AUROC = encoder treats perturbations as noise and removes them
- This is critical: if perturbation signal is washed out, the action-conditioned predictor has nothing to work with
- A good encoder should achieve AUROC > 0.6 while maintaining batch invariance

**Data requirements**: Existing data only (uses control vs case embeddings).

---

### embedding_consistency
**Implementation**: `evals/evals.py`

**Biological question**: Do replicates of the same perturbation cluster tightly in embedding space?

**Setup**: Compute L2 distances between embeddings. Compare intra-perturbation distances (same perturbation, different replicates) vs inter-perturbation distances (different perturbations).

**Metrics**:

| Metric | Description |
|--------|-------------|
| Mean Intra Distance | Average L2 distance between replicates of same perturbation |
| Mean Inter Distance | Average L2 distance between samples of different perturbations |
| Inter/Intra Ratio | Inter-perturbation / Intra-perturbation distance (higher = better) |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Inter/Intra Ratio | > 2.0 | 1.5 - 2.0 | < 1.5 | Higher = tighter within-perturbation clusters |

**Interpretation guide**:
- Ratio > 2 means replicates are well-separated from other perturbations
- Ratio near 1 means no meaningful clustering - embeddings are noisy or dominated by technical effects
- Complements batch_invariance: that eval asks "can we classify?" while this asks "do replicates cluster?"
- A model could have high classification accuracy but poor clustering if decision boundaries are messy

**Data requirements**: Existing data only.

---

### latent_space_health
**Implementation**: `evals/evals.py`

**Biological question**: Is the embedding space well-structured? Are there degenerate or collapsed dimensions?

**Setup**: Compute diagnostic statistics on the embedding space: PCA for dimensionality, variance per dimension, isotropy metrics.

**Metrics**:

| Metric | Description |
|--------|-------------|
| Effective Dim (90%) | Number of PCA components for 90% explained variance |
| Effective Dim (95%) | Number of PCA components for 95% explained variance |
| Mean/Min/Max Variance | Per-dimension variance statistics |
| N Dead Dims | Dimensions with variance < 1e-6 |
| Isotropy Ratio | Min/Max eigenvalue ratio (higher = more uniform) |
| Mean Cosine Sim | Average pairwise cosine similarity (lower = better spread) |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Effective Dim (90%) | 50-80% of D | 30-50% | < 30% or > 90% | Too low = collapsed, too high = no compression |
| N Dead Dims | 0 | 1-5 | > 5 | Dead dims indicate VICReg failure |
| Mean Cosine Sim | < 0.3 | 0.3 - 0.5 | > 0.5 | Low = good spread in space |

**Interpretation guide**:
- Dead dimensions indicate VICReg variance constraint is not working
- Very low effective dimensionality suggests representation collapse
- Very high effective dimensionality suggests no useful compression
- High mean cosine similarity suggests embeddings are clustered in a small region (not using the full space)
- This is a diagnostic eval - use it to debug training issues

**Data requirements**: Existing data only.

---

### alignment_recall
**Implementation**: `evals/evals.py`

**Biological question**: Can we retrieve the correct protein anchor from a DNA (sgRNA) query in the action space?

**Setup**: After alignment training, DNA embeddings and protein embeddings should be aligned in the action space. For each DNA perturbation, compute cosine similarity to all protein embeddings and find the rank of the correct anchor.

**Metrics**:

| Metric | Description |
|--------|-------------|
| MRR | Mean Reciprocal Rank - average of 1/rank |
| Median Rank | Typical rank of correct protein |
| Mean Rank | Average rank of correct protein |
| Recall@K | Is correct protein in top K? |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| MRR | > 0.6 | 0.3 - 0.6 | < 0.3 | 1.0 = always rank 1 |
| Recall@10 | > 0.8 | 0.5 - 0.8 | < 0.5 | Correct protein in top 10 |
| Median Rank | < 5 | 5 - 50 | > 50 | Lower is better |

**Interpretation guide**:
- High MRR means DNA and protein embeddings for the same gene are close in action space
- This is the core test of alignment training success
- Poor recall suggests the InfoNCE loss didn't align the modalities effectively
- Note: Multiple DNA sequences may map to the same protein anchor

**Data requirements**: `pert_pairs_crispri_train.npz` containing input_idx, anchor_idx pairs.

---

### modality_gap_analysis
**Implementation**: `evals/evals.py`

**Biological question**: Do DNA and Protein embeddings form separate clusters in action space, or are they well-mixed?

**Setup**: Compute action vectors for all DNA and protein embeddings. Measure centroid distance and pairwise distances within and between modalities.

**Metrics**:

| Metric | Description |
|--------|-------------|
| Centroid Distance | Euclidean distance between DNA and protein centroids |
| DNA Variance | Average squared distance from DNA centroid |
| Protein Variance | Average squared distance from protein centroid |
| Gap Ratio | Mean between-modality distance / mean within-modality distance |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Gap Ratio | < 1.5 | 1.5 - 3.0 | > 3.0 | Lower = better mixing |
| Centroid Distance | Context-dependent | - | - | Compare to within-modality spread |

**Interpretation guide**:
- Gap ratio < 1.5 means modalities overlap well in action space
- Gap ratio > 3.0 suggests alignment failed - modalities form separate clusters
- Some separation is expected due to different input dimensionalities
- The goal is that same-gene DNA/protein pairs are closer than different-gene pairs

**Data requirements**: Existing embeddings only.

---

### anchor_input_consistency
**Implementation**: `evals/evals.py`

**Biological question**: For the same target gene, do DNA (sgRNA) and Protein embeddings produce similar action vectors?

**Setup**: For each (DNA, Protein) pair targeting the same gene, compute cosine similarity between their action vectors.

**Metrics**:

| Metric | Description |
|--------|-------------|
| Mean Cosine Similarity | Average similarity between paired DNA/protein action vectors |
| Std Cosine Similarity | Spread of similarities |
| Percentiles | Distribution (5th, 25th, 50th, 75th, 95th) |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Mean Cosine Sim | > 0.8 | 0.5 - 0.8 | < 0.5 | Higher = better alignment |

**Interpretation guide**:
- High similarity means the action composer projects both modalities to similar latent actions for the same gene
- This is what enables cross-modality prediction: train on CRISPR, predict for protein perturbations
- Low similarity suggests the composer treats modalities too differently
- Tight distribution (low std) is good - all pairs should align similarly

**Data requirements**: `pert_pairs_crispri_train.npz` containing input_idx, anchor_idx pairs.

---

### mode_sensitivity
**Implementation**: `evals/evals.py`

**Biological question**: Does FiLM conditioning on perturbation mode (CRISPRi, CRISPRa, knockout) change the embeddings appropriately?

**Setup**: Pass the same DNA embeddings through the composer with different mode values. Measure how much the action vectors differ across modes.

**Metrics**:

| Metric | Description |
|--------|-------------|
| Pairwise Distances | L2 distance between same-input actions under different modes |
| Classification Accuracy | Can a linear classifier predict mode from action vector? |
| Above Chance Ratio | Accuracy / (1 / n_modes) |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Classification Accuracy | 0.4 - 0.7 | 0.35 - 0.4 or 0.7 - 0.9 | < 0.35 or > 0.9 | Moderate sensitivity is ideal |
| Above Chance Ratio | 1.2 - 2.1x | - | < 1.2x or > 2.7x | Should be meaningfully above chance but not perfect |

**Interpretation guide**:
- We want moderate mode sensitivity: the mode SHOULD affect the action vector since CRISPRi (knockdown) and CRISPRa (activation) have opposite effects
- Too low accuracy (near chance) means FiLM conditioning is not working
- Too high accuracy (near perfect) suggests mode dominates over the perturbation identity, which would hurt prediction quality
- The sweet spot is where mode contributes meaningfully but doesn't overwhelm

**Data requirements**: Existing embeddings only.

---

### target_family_probing
**Implementation**: `evals/evals.py`

**Biological question**: Do action embeddings encode protein family information about the target gene?

**Setup**: Map perturbations to their target genes, then to HGNC protein families. Train a linear classifier on action vectors to predict family membership.

**Metrics**:

| Metric | Description |
|--------|-------------|
| Accuracy | Fraction correctly classified |
| Macro F1 | F1 score averaged across families |
| Chance | 1 / n_families |
| Above Chance Ratio | Accuracy / chance |

**How to interpret**:

| Metric | Good | Average | Poor | Notes |
|--------|------|---------|------|-------|
| Above Chance Ratio | > 3x | 1.5 - 3x | < 1.5x | Higher = more family structure |
| Macro F1 | > 0.2 | 0.1 - 0.2 | < 0.1 | Depends on n_families |

**Interpretation guide**:
- High accuracy means perturbations targeting genes in the same protein family have similar action vectors
- This tests whether the action space encodes biological relationships beyond just perturbation identity
- Protein families share structural/functional properties, so similar action vectors suggest biologically meaningful embeddings
- Note: Many genes don't have family annotations, so sample size may be limited

**Data requirements**: HGNC gene family annotations from `/Users/djemec/data/jepa/reference_data/gene_family/hgnc.tsv`.

---

## Evaluations On Hold

These evaluations require additional data or architectural changes and are not currently prioritized.

### cross_cell_type_transfer
**Status**: On Hold - requires new dataset processing

**Biological question**: Does the model learn universal cell physics or K562-specific patterns?

**Setup**:
- Train on K562, evaluate zero-shot on RPE1 (or another cell line)
- Same metrics as expression_prediction
- Compare: K562->K562 vs. K562->RPE1 performance drop

**What's needed**:
- Download and process Replogle RPE1 Essential dataset from GEARS/GEO
- Run same preprocessing pipeline as K562
- Match gene sets between cell types

**Story**: If the model truly learns causal biology, some of that should transfer. Complete failure suggests overfitting to K562 idiosyncrasies. Partial transfer suggests shared biology is captured.

---

### synthetic_lethality_signal
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

### cross_modality_retrieval
**Status**: On Hold - requires chemical embeddings

**Biological question**: Given a drug embedding, can we retrieve the corresponding CRISPR knockout targeting the same pathway/mechanism?

**Setup**:
- Embed chemicals using MolBERT or ChemBERT
- For each chemical perturbation, find nearest CRISPR knockouts in action space
- Measure if compounds and knockouts targeting same genes/pathways are retrieved

**What's needed**:
- Chemical perturbation data with known targets
- Chemical embeddings (MolBERT, ChemBERT, or similar)
- Mapping between chemicals and their target genes/pathways

**Story**: This would validate that the action space truly captures biological mechanism, not just modality-specific features. A drug inhibiting gene X should have a similar action vector to CRISPR knockout of gene X.

---

## Implementation Summary

| Eval | Stage | Notebook | Biological Question | Key Metrics |
|------|-------|----------|---------------------|-------------|
| expression_prediction | Full | expression_prediction.ipynb | Predict expression after perturbation | MSE, R², Severity |
| gene_level_analysis | Full | gene_level_analysis.ipynb | Direction + DEG identification | Direction acc, Precision@K |
| perturbation_retrieval | Full | perturbation_retrieval.ipynb | Find perturbation from outcome | Recall@K, MRR |
| uncertainty_calibration | Full | uncertainty_calibration.ipynb | Are confidence estimates meaningful? | ECE, Monotonicity |
| batch_invariance | Pretrain | batch_invariance.ipynb | Batch vs biological signal | Invariance ratio |
| gene_embedding_pathways | Pretrain | gene_embedding_pathways.ipynb | Pathway structure in gene embeddings | Silhouette, k-NN |
| action_vector_pathways | Full | action_vector_pathways.ipynb | Pathway structure in action vectors | Silhouette, k-NN |
| moa_matching | Full | moa_matching.ipynb | Same-pathway similarity | Within/between ratio |
| essential_gene_prediction | Pretrain | essential_gene_prediction.ipynb | Functional importance in embeddings | Pearson r, AUROC |
| cell_type_probing | Pretrain | evals.py | Cell state disentanglement | Macro F1, Accuracy |
| reconstruction | Pretrain | evals.py | Information preservation | Pearson R, MSE |
| perturbation_detection | Pretrain | evals.py | Perturbation signal survival | AUROC |
| embedding_consistency | Pretrain | evals.py | Replicate clustering quality | Inter/Intra ratio |
| latent_space_health | Pretrain | evals.py | VICReg training diagnostics | Effective dim, Isotropy |
| alignment_recall | Alignment | evals.py | DNA-Protein retrieval | MRR, Recall@K |
| modality_gap_analysis | Alignment | evals.py | Modality separation in action space | Gap ratio |
| anchor_input_consistency | Alignment | evals.py | Same-gene cross-modality similarity | Cosine similarity |
| mode_sensitivity | Alignment | evals.py | FiLM conditioning effect | Classification acc |
| target_family_probing | Alignment | evals.py | Protein family in action embeddings | Accuracy, F1 |
| cross_cell_type_transfer | - | - | Cross-cell-type transfer | On Hold |
| synthetic_lethality_signal | - | - | Synthetic lethality detection | On Hold |
| cross_modality_retrieval | - | - | Chemical-CRISPR retrieval | On Hold (needs chemical embeddings) |

---

## Data Dependencies

| Data | Source | Evaluations | Status |
|------|--------|-------------|--------|
| K562 Essential test set | Already have | All | Done |
| Batch labels (gem_group) | Added to shards | batch_invariance | Done |
| KEGG/Reactome pathways | gseapy | gene_embedding_pathways, action_vector_pathways, moa_matching | Done |
| DepMap K562 CRISPR | DepMap API | essential_gene_prediction | Done |
| Cell type labels | Add to shards | cell_type_probing | Pending (v0.6) |
| Replogle RPE1 | GEARS/GEO | cross_cell_type_transfer | Pending |
| SynLethDB | Public database | synthetic_lethality_signal | Pending |
