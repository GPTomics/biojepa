# State of the Art: BioJEPA Evaluation Benchmarks

**Technical Reference - February 2026**

This document provides verified SOTA references for BioJEPA's 23 evaluations, with explicit comparability assessments and realistic performance targets.

**Acronyms:** DEG (Differentially Expressed Gene), MRR (Mean Reciprocal Rank), ECE (Expected Calibration Error), VEP (Variant Effect Prediction), DTI (Drug-Target Interaction)

---

## Executive Summary

### The 2026 Landscape: Key Findings

1. **Simple Baselines Often Win for Perturbation Prediction**
   > "None of the deep learning models was able to consistently outperform the mean prediction or the linear model."
   > - [Ahlmann-Eltze et al., Nature Methods, Aug 2025](https://www.nature.com/articles/s41592-025-02772-6)

2. **Metric Choice Matters More Than Model Choice**
   - Pearson on all genes (easy) vs R² on top DEGs (hard) give very different pictures
   - Most SOTA numbers use the easy metric - BioJEPA reports both

3. **Cross-Modal Perturbation Is Emerging**
   - [LPM (Nature Computational Science, 2025)](https://www.nature.com/articles/s43588-025-00870-1) is the first to unify genetic + chemical perturbations
   - BioJEPA's dual-path architecture competes in this space

4. **Many "SOTA" Models Are Not Comparable to BioJEPA**
   - Protein fitness models (AIDO, VenusREM, VespaG) solve different tasks
   - Genomic VEP models (AlphaGenome, Genos) predict variant effects on DNA
   - BioJEPA predicts perturbation-induced expression changes

### What BioJEPA Actually Does

| Stage | Task | Output |
|-------|------|--------|
| **Pretraining** | Learn cell state representations | Gene/cell embeddings |
| **Alignment** | Align perturbation embeddings (DNA/chemical -> protein target) | Action vectors |
| **Full Model** | Predict expression delta from perturbation | Delta expression profiles |

### BioJEPA's Training Datasets

| Dataset | Cell Type | Perturbation Type | Size | Notes |
|---------|-----------|-------------------|------|-------|
| **k562e_raw** | K562 | CRISPRi | ~1M cells | Replogle Essential |
| **norman** | K562 | CRISPRa | ~100K cells | Dual-gene combinatorial |
| **adamson** | K562 | CRISPRi | ~30K cells | Small-scale, GEARS benchmark |
| **sciplex** | A549, K562, MCF7 | Chemical (188 drugs) | ~600K cells | Multi-dose |
| **rep1e** | RPE1 | CRISPRi | ~1M cells | Cross-cell-type |
| **k562gw** | K562 | CRISPRi | ~2M cells | Genome-wide |

**Cell Types:** K562, RPE1, A549, MCF7, unknown (5 total)

---

## Comparability Framework

### What Makes a Valid SOTA Comparison

| Factor | Valid Comparison | Invalid Comparison |
|--------|------------------|-------------------|
| **Same task** | Expression prediction vs Expression prediction | Expression prediction vs Variant effect prediction |
| **Same dataset** | Adamson K562 vs Adamson K562 | Adamson vs ProteinGym |
| **Same metric** | Pearson (all genes) vs Pearson (all genes) | Pearson vs R² |
| **Same input** | Expression-derived embeddings vs Expression-derived | Multi-omics vs single-modality |
| **Same output** | Delta prediction vs Delta prediction | Delta vs Absolute expression |

### Models NOT Comparable to BioJEPA

The following models appear in recent SOTA discussions but solve fundamentally different tasks:

| Models | Task | Why Not Comparable |
|--------|------|-------------------|
| **AIDO Protein-RAG, VenusREM, ProteinReasoner, VespaG** | Protein fitness prediction (ProteinGym) | BioJEPA predicts expression changes, not protein fitness |
| **AlphaGenome, Genos, Evo 2** | Genomic VEP / pathogenicity | BioJEPA predicts expression deltas, not variant effects |
| **ESM3** | Protein generation | BioJEPA uses ESM-2 embeddings as input, not output |
| **PanFoMa** | Pan-cancer cell annotation | Different task (classification vs delta prediction) |

---

## Pretraining Evaluations

### 1. batch_invariance

**BioJEPA's Task:** Train linear classifier to predict batch ID from cell embeddings. A high-quality invariant embedding should make batch prediction difficult (lower accuracy = better).

**Comparability Issue:** SOTA benchmarks use scIB metrics (ASW, ARI, LISI). BioJEPA uses classifier accuracy. These are related but not identical.

| Model | Metric | Value | Comparable? | Source |
|-------|--------|-------|-------------|--------|
| **STACAS** | scIB aggregate | SOTA | No - different metric | [Nature Comm 2024](https://www.nature.com/articles/s41467-024-45240-z) |
| scANVI | scIB aggregate | 0.64 | No - different metric | scIB benchmark |
| scGPT | ASW batch | 0.92 | No - different metric | scGPT paper |
| Harmony | ASW batch | ~0.93 | No - different metric | Harmony paper |

**BioJEPA Metrics:**
- Batch classifier accuracy (lower = better)
- Above-chance ratio: accuracy / (1/n_batches)
- Perturbation classifier accuracy (should be HIGH)
- Invariance ratio: pert_acc / batch_acc (higher = better)

**Realistic Targets:**
- Batch accuracy: < 2x chance
- Perturbation accuracy: > 5x chance
- Invariance ratio: > 3

**Baseline:** Train same classifier on raw expression. BioJEPA should have LOWER batch accuracy.

---

### 2. gene_embedding_pathways

**BioJEPA's Task:** Do genes in same KEGG/Reactome pathway cluster in embedding space? Measured by silhouette score and k-NN accuracy.

| Model | Approach | Pathway Clustering | Comparable? | Source |
|-------|----------|-------------------|-------------|--------|
| **GenePT** | GPT-3.5 gene descriptions | Strong (knowledge-driven) | Partially | [bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.10.16.562533v1) |
| **scELMO** | LLM gene descriptions | Strong | Partially | scELMO paper |
| scGPT | Expression-based | Moderate (sil ~0.05-0.1) | Yes | scGPT paper |
| LPM | Perturbation-derived | Good (~0.10) | Yes | [Nature Comp Sci 2025](https://www.nature.com/articles/s43588-025-00870-1) |

**Realistic Targets:**
- Silhouette score: > 0.05 (pathways are imperfect labels - genes belong to multiple)
- k-NN accuracy: > 2x chance

**Note:** GenePT/scELMO use text descriptions, not learned from expression data. scGPT/LPM are more comparable.

---

### 3. essential_gene_prediction

**BioJEPA's Task:** Train linear probe on frozen gene embeddings to predict DepMap K562 CRISPR essentiality scores.

**Comparability Issue:** Dedicated models use multi-omics features. BioJEPA uses only expression-derived embeddings.

| Model | Input Features | AUROC | Comparable? | Source |
|-------|----------------|-------|-------------|--------|
| **DeEPsnap** | Multi-omics (seq, PPI, GO, expr) | **96.16%** | No - more inputs | [Sci Reports Jul 2025](https://www.nature.com/articles/s41598-025-99164-9) |
| DeepHE | Sequence + network | 91-92% | No - more inputs | DeepHE paper |
| Bingo | LLM + GNN | Good | No - more inputs | Bingo paper |
| **Elastic Net** | Expression only | ~65-70% | **Yes** | Baseline |

**Realistic Targets:**
- AUROC: > 0.65
- Pearson r: > 0.2

**Why Lower Than DeEPsnap?** BioJEPA uses only expression-derived embeddings. Multi-omics models have sequence, network, and ontology features that encode essentiality more directly.

---

### 4. cell_type_probing

**BioJEPA's Task:** Classify cells by cell type using mean-pooled embeddings.

**Comparability Issue:** BioJEPA has only 5 cell types. SOTA benchmarks use 10-50+ types.

| Model | Dataset | Cell Types | Accuracy | Comparable? | Source |
|-------|---------|------------|----------|-------------|--------|
| scGPT (fine-tuned) | Pancreas | ~15 | 91% | No - more classes | scGPT paper |
| **PanFoMa** | Pan-cancer | 33+ | 93% | No - more classes | [arXiv 2025](https://arxiv.org/html/2512.03111v1) |
| GeneFormer | Various | Varies | 64-85% | No - more classes | GeneFormer paper |
| **Any classifier** | BioJEPA's 5 types | 5 | ? | **Yes** | - |

**Realistic Targets:**
- Macro F1: > 0.7
- Above chance ratio: > 4x

**Note:** With only 5 cell types (K562, RPE1, A549, MCF7, unknown), this is a simpler task. Focus on beating random baseline significantly.

---

### 5-8. Other Pretraining Evals

| Eval | Task | Baseline | Target | Notes |
|------|------|----------|--------|-------|
| **reconstruction** | Reconstruct expression from embeddings | Linear on raw | Pearson > 0.7 | Sanity check |
| **perturbation_detection** | Classify control vs perturbed | Random (0.5) | AUROC > 0.6 | Perturbation signal survives encoding |
| **embedding_consistency** | Replicates cluster together | Random | Inter/Intra > 1.8 | Biological replicability |
| **latent_space_health** | Check for collapsed dimensions | Degenerate | 0 dead dims, eff dim 50-80% | Training diagnostic |

---

## Alignment Evaluations

### 9. seq_to_target_retrieval

**BioJEPA's Task:** Given a sequence embedding (DNA or chemical), retrieve the correct protein target. Measured by MRR and Recall@K.

**Comparability Issue:** Most DTI models predict interaction probability (classification, AUROC). BioJEPA does retrieval (ranking, MRR).

| Model | Task | Metric | Value | Comparable? | Source |
|-------|------|--------|-------|-------------|--------|
| **SP-DTI** | DTI prediction | AUROC | 0.873 | No - classification not retrieval | SP-DTI paper |
| GraphDTA | DTI prediction | AUROC | ~0.78 | No - different task | GraphDTA paper |
| **LPM** | Cross-modal retrieval | Recall | High | **Partially** | [Nature Comp Sci 2025](https://www.nature.com/articles/s43588-025-00870-1) |

**Realistic Targets:**
- DNA retrieval: MRR > 0.6, Recall@10 > 0.8 (alignment objective)
- Chemical retrieval: MRR > 0.3, Recall@10 > 0.5 (less data)

**Baseline:** Random retrieval MRR ≈ 1/N_targets ≈ 0.001

---

### 10. cross_modality_target_consistency

**BioJEPA's Task:** Do different DNA sequences targeting the same gene produce similar action vectors?

**Note:** Despite the name, this eval tests DNA→DNA consistency (multiple sgRNAs targeting same gene), not DNA↔chemical.

| Model | Approach | Within/Between Ratio | Source |
|-------|----------|---------------------|--------|
| **LPM** | Unified embedding | High (CRISPR↔drug clusters) | [Nature Comp Sci 2025](https://www.nature.com/articles/s43588-025-00870-1) |
| Random | No alignment | ~1.0 | Baseline |

**Realistic Target:** Within/Between ratio > 1.3

---

### 11-17. Other Alignment Evals

| Eval | Task | SOTA Reference | Target | Notes |
|------|------|----------------|--------|-------|
| **seq_target_gap_analysis** | Measure space gap per modality | LPM unified space | Gap ratio < 2.0 | Per-modality breakdown |
| **paired_alignment_quality** | Cosine sim for known pairs | Alignment objective | Mean cosine > 0.5 | Direct alignment measure |
| **mode_sensitivity** | Does FiLM differentiate modes? | BioJEPA-specific | Accuracy 0.3-0.6 | Tests 7 of 9 modes |
| **fusion_quality** | Is fused > seq-only + target-only? | CPA combo effects | Fused variance >= max(seq,target) | Tests fusion MLP |
| **missing_data_robustness** | Performance with missing modality | LPM P-R-C | Retention > 0.6 | Graceful degradation |
| **multi_pert_alignment** | Dual-gene alignment quality | Norman data | < 25% degradation | Uses real multi-pert samples |
| **target_family_probing** | Protein family classification | ESM-2 baseline | Above chance > 2x | Tests biological structure |

**Note:** Many alignment evals are BioJEPA-specific with no external benchmarks. Focus on internal baselines.

---

## Full Model Evaluations

### 18. expression_prediction

**BioJEPA's Task:** Predict gene expression **DELTA** (change from control) after perturbation.

**CRITICAL Comparability Issues:**

1. **Delta vs Absolute:** BioJEPA predicts deltas. Some models predict absolute expression.
2. **Metric Choice:**
   - Pearson on ALL genes (easy): Most genes don't change
   - Pearson/R² on TOP DEGs (hard): The real test
3. **Dataset Differences:** Numbers only comparable on same dataset

#### SOTA on Adamson K562 (Pearson, All Genes)

*All numbers verified against original sources*

| Model | Pearson | Notes | Source |
|-------|---------|-------|--------|
| **scLAMBDA** | **0.786** | Deep generative, 10 replicates, top 5K HVGs | [bioRxiv/PMC Dec 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11643044/) |
| **GenePert** | **0.79** | GPT-4 embeddings | [bioRxiv Oct 2024](https://www.biorxiv.org/content/10.1101/2024.10.27.620513v1) |
| GEARS | 0.692 | Graph + GO features | [Nature Biotech 2023](https://www.nature.com/articles/s41587-023-01905-6) |
| scGPT | 0.661 | Foundation model | [Nature Methods 2024](https://www.nature.com/articles/s41592-024-02201-0) |

#### SOTA on Replogle K562 (Pearson, All Genes)

*Source: [BMC Genomics benchmark, Apr 2025](https://link.springer.com/article/10.1186/s12864-025-11600-2)*

| Model | Pearson | Notes |
|-------|---------|-------|
| RF + GO features | **0.480** | Simple ML beats foundation models |
| **Train Mean (baseline)** | **0.373** | Just predict training mean |
| scGPT | 0.327 | Underperforms baseline |
| scFoundation | 0.269 | Underperforms baseline |

**Key Insight:** On Replogle K562, simple baselines beat foundation models.

#### What BioJEPA Reports (Different Metrics)

| Metric | What It Measures | Target | Notes |
|--------|------------------|--------|-------|
| Global MSE | Overall error | Lower is better | Scale-dependent |
| Pearson R (Top 20 DEGs) | Correlation on hardest genes | > 0.5 | Hard metric |
| R² (All Genes) | Variance explained | > 0.7 | Easy metric (most genes unchanged) |
| R² (Top 50 DEGs) | Variance on DEGs | > 0.0 | Hardest metric (negative = worse than mean) |
| Severity Pearson | Perturbation magnitude ranking | > 0.4 | Can model rank effect sizes? |

**Baseline:** Predict zero delta for all genes. R² on top DEGs will be negative.

---

### 19. gene_level_analysis

**BioJEPA's Task:** Predict direction (UP/DOWN/UNCHANGED) and identify top DEGs.

| Model | Task | Metric | Value | Source |
|-------|------|--------|-------|--------|
| **SynthPert** | 3-class direction | AUROC | **78%** (PerturbQA) | [arXiv 2025](https://arxiv.org/html/2509.25346v1) |
| scGPT | Top DEG recovery | Precision | Inflated | Note: predicts target gene itself |

**Realistic Targets:**
- Direction Accuracy (Top 50): > 0.55 (chance = 0.33)
- Precision@20: > 0.2 (chance ≈ 0.016)
- NDCG@20: > 0.4

---

### 20. perturbation_retrieval

**BioJEPA's Task:** Given a desired expression change, find which perturbation causes it.

| Model | Approach | Metric | Comparable? | Source |
|-------|----------|--------|-------------|--------|
| **CIGER** | Deep ranking | Best on drug retrieval | Partially | [Patterns Feb 2022](https://www.sciencedirect.com/science/article/pii/S2666389922000149) |
| **CPA** | Latent matching (no adversarial) | Best on rank metrics | **Yes** | [MSB Jun 2023](https://www.embopress.org/doi/abs/10.15252/msb.202211517) |
| Connectivity Map | Signature correlation | Baseline | Yes | CMap paper |

**Realistic Targets (with ~1000 perturbations):**
- Recall@10: > 0.4
- MRR: > 0.25

**Baseline:** Random Recall@10 ≈ 0.01, MRR ≈ 0.001

---

### 21. uncertainty_calibration

**BioJEPA's Task:** Does predicted uncertainty (Gaussian NLL) correlate with actual error?

| Model | Approach | Notes | Source |
|-------|----------|-------|--------|
| **TISSUE** | Conformal inference | Formal coverage guarantees, 90%+ | [Nature Methods Feb 2024](https://www.nature.com/articles/s41592-024-02184-y) |
| Deep Ensembles | Multiple networks | Strong but expensive | Standard |
| MC Dropout | Single network | Easy to implement | Standard |

**Realistic Targets:**
- Uncertainty-Error Pearson: > 0.15 (positive = informative)
- ECE: < 0.2
- Monotonicity: > 60%

**Note:** BioJEPA's uncertainty is in latent space (not gene space), so correlations may be modest.

---

### 22-23. action_vector_pathways & moa_matching

**BioJEPA's Task:** Do same-pathway perturbations have similar action vectors / predicted effects?

| Model | Approach | Performance | Source |
|-------|----------|-------------|--------|
| **LPM** | Perturbation embeddings | Good pathway clustering | [Nature Comp Sci 2025](https://www.nature.com/articles/s43588-025-00870-1) |
| Raw signatures | Expression correlation | Moderate (noise-limited) | Baseline |

**Realistic Targets:**
- Silhouette (KEGG): > 0.05
- Within/Between ratio: > 1.2
- p-value: < 0.05

---

## BioJEPA's Unique Capabilities

Features that distinguish BioJEPA from existing models:

| Capability | BioJEPA | GEARS | scGPT | CPA | LPM |
|------------|---------|-------|-------|-----|-----|
| **Multi-Perturbation (up to 4)** | Yes | 2 only | No | Yes | Yes |
| **Dual-Path Fusion (seq+target)** | Yes | No | No | No | No |
| **Mode Conditioning (9 types)** | Yes (FiLM) | No | No | Dose only | Yes |
| **Uncertainty (Gaussian NLL)** | Yes | No | No | No | No |
| **Missing Data Fallback** | Yes | No | No | No | Yes |
| **Cross-Modal (DNA + Chemical)** | Yes | No | No | Yes | Yes |

These capabilities are tested by BioJEPA-specific evals (mode_sensitivity, fusion_quality, missing_data_robustness) that don't have external benchmarks.

---

## Summary: Realistic Performance Targets

### Pretraining Evals

| Eval | Metric | Target | Baseline |
|------|--------|--------|----------|
| **batch_invariance** | Invariance ratio | > 3 | ~1 (no separation) |
| **gene_embedding_pathways** | Silhouette | > 0.05 | ~0 (random) |
| **essential_gene_prediction** | AUROC | > 0.65 | 0.5 (random) |
| **cell_type_probing** | Macro F1 | > 0.7 | 0.2 (random with 5 classes) |
| **reconstruction** | Pearson R | > 0.7 | - |
| **perturbation_detection** | AUROC | > 0.6 | 0.5 (random) |
| **embedding_consistency** | Inter/Intra | > 1.8 | ~1 (random) |
| **latent_space_health** | Dead dims | 0 | - |

### Alignment Evals

| Eval | Metric | Target | Baseline |
|------|--------|--------|----------|
| **seq_to_target_retrieval (DNA)** | MRR | > 0.6 | ~0.001 (random) |
| **seq_to_target_retrieval (chem)** | MRR | > 0.3 | ~0.001 (random) |
| **cross_modality_target_consistency** | Within/Between | > 1.3 | ~1 (random) |
| **paired_alignment_quality** | Mean cosine | > 0.5 | ~0 (random) |
| **mode_sensitivity** | Accuracy | 0.3-0.6 | 0.14 (random with 7 tested modes) |
| **fusion_quality** | Variance ratio | > 1.0 | - |
| **missing_data_robustness** | Retention | > 0.6 | - |

### Full Model Evals

| Eval | Easy Metric | Hard Metric | Target (Hard) |
|------|-------------|-------------|---------------|
| **expression_prediction** | Pearson (all) > 0.6 | R² (top 50 DEGs) | > 0.0 |
| **gene_level_analysis** | Direction (all) > 0.7 | Precision@20 | > 0.2 |
| **perturbation_retrieval** | - | MRR | > 0.25 |
| **uncertainty_calibration** | - | Error-Unc Pearson | > 0.15 |
| **action_vector_pathways** | - | Silhouette | > 0.05 |
| **moa_matching** | - | Similarity ratio | > 1.2 |

---

## Recommended Evaluation Strategy

### 1. Establish Internal Baselines

For each eval, compute:
- **Random baseline:** Random embeddings/predictions
- **Simple baseline:** Mean prediction, linear on raw expression
- **Input baseline:** Raw embeddings (NT, ESM-2, ChemMRL) without BioJEPA training

### 2. Compare Within-Dataset

Don't compare Adamson numbers to Replogle numbers. Compare:
- BioJEPA on k562e_raw (= Replogle K562 Essential) vs GEARS/scGPT on same dataset
- BioJEPA on Norman vs published Norman benchmarks
- BioJEPA on Adamson vs scLAMBDA/GenePert on Adamson

### 3. Use Correct Metrics

If comparing to SOTA:
- Match the metric exactly (Pearson all-genes vs Pearson top-20 DEGs)
- Note if task is delta vs absolute prediction
- Report both easy AND hard metrics

### 4. Report BioJEPA-Specific Evals Separately

For mode_sensitivity, fusion_quality, multi_pert_alignment, etc.:
- No external comparison available
- Report improvement over internal baselines
- Verify the capability functions as designed

---

## Verified Sources

### Primary Benchmark Studies
- [Deep learning does not outperform baselines - Nature Methods (Aug 2025)](https://www.nature.com/articles/s41592-025-02772-6) - Ahlmann-Eltze, Huber, Anders. Vol 22, pp 1657-1661
- [Benchmarking foundation cell models - BMC Genomics (Apr 2025)](https://link.springer.com/article/10.1186/s12864-025-11600-2) - Source for Replogle K562 numbers (0.373, 0.327, 0.269)
- [LPM - Nature Computational Science (2025)](https://www.nature.com/articles/s43588-025-00870-1) - First unified genetic + chemical perturbation model

### Expression Prediction Models
- [scLAMBDA - bioRxiv/PMC (Dec 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11643044/) - Yale, 0.786 Pearson on Adamson
- [GenePert - bioRxiv (Oct 2024)](https://www.biorxiv.org/content/10.1101/2024.10.27.620513v1) - Zou group, GPT-4 embeddings, 0.79 on Adamson
- [GEARS - Nature Biotechnology (2023)](https://www.nature.com/articles/s41587-023-01905-6) - Graph + GO, 0.692 on Adamson
- [scGPT - Nature Methods (2024)](https://www.nature.com/articles/s41592-024-02201-0) - Foundation model, 0.661 on Adamson

### Other Models
- [CPA - Molecular Systems Biology (Jun 2023)](https://www.embopress.org/doi/abs/10.15252/msb.202211517) - Lotfollahi et al., perturbation combinations
- [SynthPert - arXiv (2025)](https://arxiv.org/html/2509.25346v1) - LLM reasoning traces, 78% AUROC on direction
- [TISSUE - Nature Methods (Feb 2024)](https://www.nature.com/articles/s41592-024-02184-y) - Conformal inference for uncertainty
- [STACAS - Nature Communications (Jan 2024)](https://www.nature.com/articles/s41467-024-45240-z) - Semi-supervised batch integration
- [DeEPsnap - Scientific Reports (Jul 2025)](https://www.nature.com/articles/s41598-025-99164-9) - 96.16% AUROC for essential genes (multi-omics)
- [CIGER - Patterns (Feb 2022)](https://www.sciencedirect.com/science/article/pii/S2666389922000149) - Chemical-induced gene expression ranking

### Datasets
- [Replogle et al. - Cell (2022)](https://www.cell.com/cell/fulltext/S0092-8674(22)00597-9) - K562 genome-wide CRISPRi
- [Norman et al. - Science (2019)](https://www.science.org/doi/10.1126/science.aax4438) - Dual-gene CRISPRa

### Not Comparable (Different Tasks)
- [PanFoMa - arXiv (2025)](https://arxiv.org/html/2512.03111v1) - Pan-cancer cell annotation (Mamba architecture)
- [AIDO Protein-RAG - ProteinGym](https://proteingym.org/benchmarks) - Protein fitness prediction
- [AlphaGenome - Nature (Jan 2026)](https://pubmed.ncbi.nlm.nih.gov/41606153/) - Genomic VEP with 1Mb context
