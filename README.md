**In progress version:**  v0.6
## V0.4 Technical Report [AVAILABLE](https://github.com/GPTomics/biojepa/blob/main/docs/technical_report_v0_4.pdf)

# bio-JEPA

Our goal is to build a "world model" for cells as inspired by [V JEPA 2-AC](https://github.com/facebookresearch/vjepa2). This means that a successful model learns the causal physics of cell states. Because there are billions of potential drug and gene combinations, it is impossible and too expensive for scientists to test them all in a physical lab to see what works. A successful model would act like a digital simulator that predicts the results of these experiments instantly, allowing prediction of how cell types would react to different perturbations (e.g. therapeutics, gene knockout).

## Background

Inspired by [V JEPA 2-AC](https://github.com/facebookresearch/vjepa2), we treat perturbation response as an action-conditioned latent prediction problem. V-JEPA learns to predict future video frames given abstract "actions" inferred from data. We apply the same principle to cells: the perturbation is the action, and the model learns to predict the resulting gene expression state in a compressed latent space that captures pathway and regulatory network relationships.

### Target Corporate Uses

**Use Case 1: Target Discovery** based on a gene signature.  Different teams around the world have a wide range of molecular disease signatures that they do not know how to target yet. An example could be having the signature of a specific chemotherapy resistant cancer. A common question is: "what would need to be done to these cells to add sensitivity back in".  If you wanted to do this in a wet lab you'd maybe run CRISPR based screens taking months of time and hundreds of thousands of dollars.  *How this model helps:* With a model you can create an encoding of a known version of the sensitive cell and one of the resistant cell. From this, run a series of input optimization steps to find the vector that minimizes the distance between the sensitive and resistant cell then map that vector back to a perturbation (or a couple of them). After this, the wetlab experimentation can be focused on the in-silico recommendation. 

**Use Case 2: Lead Optimization** by acting as a safety oracle. If a team has a promising candidate but they haven't yet run safety profiling (e.g. what is the toxocitiy specifically in the heart or liver).  In this example, the model understands baseline expression for different cell lines already so the user would create an embedding for the new drug (acting as the perturbation) and then pass it through the model to see the mechanism impacts on various cell types.  Based on bioloical understanding, or a finetuned head,  a user can then understand impact on the different virtual cells and can try out minor tweaks to the drug to improve the safety. 

**Use Case 3: Drug Combination** by establishing the effects of combined perturbations.  Many cancers are not cured with a single drug but knowing the right combination of drugs is extremely difficult the more there are out there. By having a model that understands baseline cell behavior and the cellular impact from the common standard-of-care regiments, a team could then use the model to test pairwise combination between the standard of care and/or with novel therapeutics.  Being able to screen these combinations and see the combined impact, teams can find the best combinations. Alternatively, a user can analyze the standard of care comapred to a healthy control cell, derive the action vector (similar to use case 1) that minimzes the difference, and then convert that backwards into a known perturbation impact.

**Use Case 4: Mechanism of Action Inference** by comparing unknown compounds to known perturbations. When a team discovers a compound with interesting phenotypic effects but doesn't understand how it works, they traditionally run expensive pull-down assays or genetic screens to identify the target. *How this model helps:* By embedding the compound's expression signature and comparing it to the learned action vectors of known genetic perturbations, the model can identify which gene knockdowns or knockups produce the most similar cellular response. If a novel compound's effect closely resembles a KRAS knockdown, that suggests KRAS pathway involvement without any wet-lab target identification.

**Use Case 5: Synthetic Lethality Prediction** by identifying gene pairs that are lethal when both perturbed. Cancer cells often have mutations that make them uniquely vulnerable to specific secondary perturbations that would be tolerable in healthy cells. Finding these synthetic lethal pairs is traditionally done through exhaustive combinatorial screens. *How this model helps:* By learning the causal physics of gene interactions, the model can predict which combinations of perturbations would push a cell into a non-viable state. A user could systematically query pairs of perturbations against a cancer cell embedding and flag combinations where the predicted state diverges dramatically from viability, prioritizing candidates for experimental validation.

**Use Case 6: Resistance Prediction** by anticipating how cells escape therapeutic pressure. A major challenge in cancer treatment is that tumors evolve resistance to initially effective drugs, often through predictable compensatory mechanisms. Understanding these escape routes early could inform combination strategies that cut off resistance before it emerges. *How this model helps:* Given a perturbation and a cell state, the model can be queried to identify which secondary expression changes would counteract the perturbation's effect. By analyzing the latent space, a user could map the "escape trajectories" and identify the genes or pathways most likely to be upregulated in resistant populations, then design combinations that block those routes.

**Use Case 7: Cell State Engineering** by finding perturbation sequences that drive cells toward target states. In cell therapy manufacturing (iPSC differentiation, CAR-T production), the goal is to reliably transform cells from one state to another. Current protocols are discovered through trial-and-error and often have low efficiency or batch variability. *How this model helps:* A user can embed both the starting cell state and the desired target state, then use the model to search for perturbations (or sequences of perturbations) that minimize the distance between the predicted outcome and the target. This turns protocol optimization into a search problem over the model's learned action space rather than exhaustive empirical testing.

**Use Case 8: Biomarker Discovery** by identifying genes that predict perturbation response. Patient stratification is critical for precision medicine—knowing which patients will respond to a therapy before treating them. Currently this requires large clinical trials with post-hoc biomarker analysis. *How this model helps:* By analyzing which features of the control cell state most strongly influence the predicted response magnitude for a given perturbation, the model can surface candidate biomarkers. Genes whose baseline expression levels correlate with large predicted shifts under a specific drug become candidates for patient selection markers, testable in smaller focused trials.

**Use Case 9: Experiment Prioritization** by guiding which perturbations to test next. Wet-lab perturbation screens are expensive and time-consuming, so choosing which experiments to run matters. Random or brute-force screening wastes resources on uninformative experiments. *How this model helps:* The model's uncertainty estimates indicate where its predictions are least confident—these are the perturbations where experimental data would be most valuable for improving the model. A team could use this as an active learning loop: run the model, identify high-uncertainty perturbations, test those in the lab, retrain, and repeat. This focuses experimental budgets on the experiments that maximally improve the model's coverage.

### Target Research Uses

Beyond therapeutics, the model's learned representations enable biological discovery:

**System-Level Discovery:** Analyzing attention maps can uncover hidden regulatory relationships between pathways (e.g., identifying that Hypoxia causally triggers DNA Repair machinery).

**New Mechanism Discovery:** If the model requires a previously uncharacterized gene/protein to make accurate predictions, this provides evidence that the component is functionally essential to specific biological processes.

## Current Performance

| Metric | v0.5 | v0.4 | v0.3 | v0.2 | Context |
| - | - | - | - | - | - |
| **Global MSE** | **0.489** | 0.498 | 0.515 | 0.790 | Lower is better |
| **Pearson R (Top 20)** | 0.919 | **0.927** | 0.921 | 0.605 | On genes that change most |
| **R² All (mean/median)** | 0.930 / 0.940 | 0.918 / 0.927 | 0.902 / 0.910 | **0.942 / 0.956** | Inflated metric (most genes don't change) |
| **R² Top 50 (mean/median)** | 0.066 / **0.341** | **0.096** / 0.325 | 0.060 / 0.255 | -0.027 / 0.269 | The hard test - no published SOTA |
| Severity Correlation | 0.835 | **0.870** | — | — | Predicting effect magnitude |
| Direction Accuracy (All) | **89.7%** | 87.7% | — | — | UP/DOWN/UNCHANGED classification |
| Direction Accuracy (Top 50) | 28.5% | **34.0%** | — | — | Harder - on genes that change |
| DEG Precision@20 | **4.8%** | 3.1% | — | — | Random baseline ~0.4% |
| DEG vs Random | **12.0x** | 7.8x | — | — | Improvement over chance |
| Retrieval MRR | 0.007 | **0.010** | — | — | ~1250 perturbations to rank |
| Retrieval Recall@10 | **3.0%** | 2.0% | — | — | Random = 0.8% |
| Uncertainty ECE | 0.281 | **0.135** | — | — | Lower is better |
| Uncertainty Monotonicity | 0.56 | **0.78** | — | — | Higher is better |
| Batch Invariance Ratio | 0.167 | **0.215** | — | — | Higher = more bio vs technical |
| Pathway Gene k-NN | **30.7%** | 27.4% | — | — | Pathway structure in gene embeddings |
| Pathway Action k-NN | **41.0%** | 37.9% | — | — | Pathway structure in action vectors |
| MOA Similarity Ratio | 1.005 | **1.006** | — | — | >1 = same-pathway more similar |
| Essential AUROC | **0.741** | 0.707 | — | — | Predicting gene essentiality |
| Essential Pearson | **0.408** | 0.275 | — | — | Correlation with DepMap scores |

Extended evals (rows without bold) available for v0.4+.

**Notes:** R² on all genes will always look good because ~95% of genes barely change. The real test is R² on Top 50 DEGs. Most published SOTA uses Pearson on all genes (easy metric). Retrieval metrics are against ~1250 perturbations. For detailed metric interpretation, see `docs/eval_planning.md`.

### SOTA Context

**Published SOTA (Pearson, all genes, Adamson K562):**
| Model | Pearson | Notes |
|-------|---------|-------|
| scLAMBDA | 0.786 | Current SOTA |
| GenePert | 0.79 | GPT-4 embeddings |
| GEARS | 0.692 | Graph + GO |

**Key finding:** Simple baselines often beat foundation models. On Replogle K562, a train-mean baseline (0.373 Pearson) outperforms scGPT (0.327). See [Ahlmann-Eltze et al., Nature Methods 2025](https://www.nature.com/articles/s41592-025-02772-6).

For detailed SOTA analysis, see `docs/sota_evals.md`.

### What Makes BioJEPA Different

| Capability | BioJEPA | GEARS | scLAMBDA | LPM |
|------------|---------|-------|----------|-----|
| Genetic perturbations | Yes | Yes | Yes | Yes |
| Chemical perturbations | Yes | No | No | Yes |
| Multi-pert (>1 simultaneous) | Up to 4 | 2 (graph) | No | No |
| Uncertainty quantification | Yes (Gaussian NLL) | No | No | No |
| Missing data fallback | Yes | N/A | N/A | Partial |
| Mode conditioning | 9 modes (FiLM) | No | No | No |

## v0.6 Architecture (In Progress)

**Training:** 6 datasets (~5.5M cells, see Datasets table below), 16,384 genes. Config: embd=256, heads=4, layers=6, lat_dim=320, mod_dim=64, max_perts=4.

1. **Dual-Pathway Perturbation Encoding with Sequence-Target Fusion:** The ActionComposer separately encodes the perturbation sequence (sgRNA, protein, or chemical) and its biological target (protein), then fuses them via concat+MLP when both are available. Sequences use modality-specific projectors (DNA via [NucleotideTransformer](https://huggingface.co/InstaDeepAI/NTv3_650M_pre): 1536->D, protein via [ESM-2](https://huggingface.co/facebook/esm2_t6_8M_UR50D): 320->D, chemical via [ChemMRL](https://huggingface.co/Derify/ChemMRL): 1024->D). When only one input is available, it passes through directly; a learned unknown embedding handles missing data.
    1. Benefit: Cleanly separates "what is perturbing" (sequence) from "what is being perturbed" (target), enabling alignment training between sequences and targets while gracefully handling variable data availability across perturbation types.
2. **Multi-Perturbation Support via Cross-Attention:** The model handles up to 4 simultaneous perturbations per sample. Each perturbation produces an action token, and the ACPredictor cross-attends to all action tokens when predicting the perturbed cell state.
    1. Benefit: Enables modeling of combinatorial perturbations (dual CRISPRi, drug combinations) and learning non-linear interaction effects between perturbations.
3. **Expanded Mode Vocabulary:** Mode conditioning now covers genetic and chemical perturbation types: CRISPRi (0), CRISPRa (1), overexpression (2), knockout (3), inhibitor (4), agonist (5), degrader (6), binder (7), unknown (8). Mode is applied via FiLM conditioning after sequence-target fusion.
    1. Benefit: Encodes that the same target can be affected differently depending on the perturbation mechanism (e.g., CRISPRi knockdown vs small molecule inhibitor targeting the same protein).
4. **Attention Pooling for Alignment:** A learned attention pooling mechanism combines multiple action vectors into a single vector for contrastive alignment training. A learned query attends to all perturbation tokens with masking for variable-length inputs.
    1. Benefit: Enables alignment training on multi-perturbation samples while learning which perturbations are most informative for alignment.
5. **RMSNorm Normalization:** Replaced LayerNorm with RMSNorm throughout the CellStateEncoder, MaskedPredictor, and ACPredictor. RMSNorm simplifies normalization by removing mean-centering, using only root-mean-square scaling. Implementation uses FP32 upcast for numerical stability during mixed-precision training.
    1. Benefit: Follows modern transformer best practices (Llama 3, Qwen3, DeepSeek V3, Gemma 3). Reduces parameters, improves compute efficiency, and provides equivalent or better training stability.
6. **SwiGLU Feed-Forward Networks:** Replaced GELU MLP with SwiGLU (Swish-Gated Linear Unit) in all transformer blocks. Uses three weight matrices with the gating pattern `w3(silu(w1(x)) * w2(x))`, bias=False, and hidden dimension set to `embed_dim * mlp_ratio * 2/3` rounded to multiples of 64 for tensor core efficiency.
    1. Benefit: Multiplicative gating improves gradient flow and model expressivity. The gating mechanism has biological relevance as gene regulation inherently involves activation/inhibition gating.
7. **Output Gating for Linear Attention:** Added a learnable sigmoid gate to the linear attention output in BioLinearAttention. The gate is computed from the query input (`sigmoid(gate(x))`) and multiplies the attention output before the final projection. Critically, gating is applied only to self-attention (`kv is None`), not cross-attention, to preserve the perturbation response pathway in the ACPredictor.
    1. Benefit: Provides per-gene control over attention contribution, allowing the model to learn which genes should dominate attention patterns. Proven effective for linear attention architectures (Gated DeltaNet, Qwen3-Next, Kimi K2).

## v0.5 Architecture

**Training:** [Gears K562](https://maayanlab.cloud/Harmonizome/dataset/Replogle+et+al.%2C+Cell%2C+2022+K562+Essential+Perturb-seq+Gene+Perturbation+Signatures), PT:100 Ep, Tr:20 Ep, Dec:20 Ep. Params: JEPA 6M, AC 7.9M, Pert 883K. Config: embd=256, heads=4, layers=6, lat_dim=320, mod_dim=64.

1. **Dual-Pathway Expression Value Encoding:** Replaced the simple scaled projection in the CellStateEncoder with a dual-pathway system combining linear scaling and Gaussian Fourier Projection, fused via FiLM-based modulation.
    1. Benefit: Better captures multi-scale properties of gene expression values. The Fourier pathway provides frequency-based representation of expression magnitudes while the linear pathway preserves direct scaling relationships. FiLM fusion allows the model to learn optimal blending between pathways.

## v0.4 Architecture

**Training:** [Gears K562](https://maayanlab.cloud/Harmonizome/dataset/Replogle+et+al.%2C+Cell%2C+2022+K562+Essential+Perturb-seq+Gene+Perturbation+Signatures), PT:100 Ep, Tr:20 Ep, Dec:20 Ep. Params: JEPA 6M, AC 7.9M, Pert 883K. Config: embd=256, heads=4, layers=6, lat_dim=320, mod_dim=64.

1. **Unified, Modular Perturbation Encoding with Explicit Target Awareness:** Perturbations are now represented as composite inputs that include both the perturbation itself and its biological target. For example, CRISPRi perturbations embed the sgRNA sequence (via  [Nucleotide Transformer v3_650M_pre](https://huggingface.co/InstaDeepAI/NTv3_650M_pre)) alongside the target gene’s protein sequence (via [ESM-2 8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D)). These heterogeneous embeddings are projected through modality-specific encoders (DNA, protein, chemical) and fused by a FiLM-based perturbation composer conditioned on perturbation mode (e.g., CRISPRi, CRISPRa, overexpression).
    1. Benefit: Establishes a general perturbation abstraction that cleanly separates what is perturbed, what it targets, and how it acts, enabling zero-shot generalization across modalities and perturbation types while keeping the core cell model agnostic to featurization details.
2. **Standardized Perturbation Conditioning via Modality and Mode Tokens:** Each perturbation is explicitly annotated with modality identifiers (protein, DNA, chemical) and mode identifiers (CRISPRi, CRISPRa, overexpression, control, etc.), which are used by the composer during conditioning.
    1. Benefit: Provides a principled, extensible conditioning interface that avoids hard-coded assumptions in the embedding space and supports systematic expansion to new perturbation classes and mechanisms.
3. **Decoupled Perturbation Representations from the Core BioJEPA Model:** The core BioJEPA architecture no longer owns perturbation embeddings. Instead, perturbation representations are supplied externally via embedding banks and processed by the composer at runtime.
    1. Benefit: Cleanly separates cell-state representation learning from perturbation featurization, allowing perturbation encoders to be swapped, upgraded, or scaled independently without retraining or bloating the core model.

## v0.3 Architecture

**Training:** [Gears K562](https://maayanlab.cloud/Harmonizome/dataset/Replogle+et+al.%2C+Cell%2C+2022+K562+Essential+Perturb-seq+Gene+Perturbation+Signatures), PT:100 Ep, Tr:20 Ep, Dec:20 Ep. Params: JEPA 6M, AC 7.9M. Config: embd=256, heads=4, layers=6.

1. **Replaced Attention Mechanism:** Switched to linear attention (using kernelized attention with ELU + 1 feature maps).
    1. Benefit: Reduces complexity from $O(N^2)$ to $O(N)$ 
2. **Removed Explicit Pathway Layer:** Removed the fixed pathway weights. The encoder now processes gene embeddings directly rather than projecting them into defined biological pathways.
    1. Benefit: Removes human bias and rigid sparsity constraints, allowing the model to learn latent gene-gene relationships and pathway definitions purely from data.
3. **New Loss Functions:** Implemented Variance-Invariance-Covariance Regularization (VICReg) alongside reconstruction. Pretraining uses L1 + VICReg; Prediction uses Gaussian NLL + VICReg.
    1. Benefit: Prevents "posterior collapse" (where all embeddings look the same) and forces the latent dimensions to be statistically independent and information-rich.
4. **Probabilistic Predictor Output:** The ACPredictor now generates a distribution of possible outcomes (mean and variance) rather than a single fixed number.
    1. Benefit: Allows the model to capture biological noise and express uncertainty in its predictions, preventing it from hallucinating precision where none exists.
5. **Overhauled Predictor Conditioning:** Removed Adaptive Layer Norm (AdaLN) in the ACPredictor and now injects action information via Cross-Attention.
    1. Benefit: Provides a more expressive mechanism for the perturbation to influence the cell state updates directly in the residual stream, rather than just scaling normalization statistics.
6. **Query Mechanism Update:** The ACpredictor now generates queries based on target indices via an embedding layer, rather than concatenating a fixed sequence of learnable mask tokens.
    1. Benefit: Explicitly signals to the model which specific gene targets it needs to reconstruct, improving prediction accuracy over implicit positional learning.

## v0.2 Architecture

**Training:** [Gears K562](https://maayanlab.cloud/Harmonizome/dataset/Replogle+et+al.%2C+Cell%2C+2022+K562+Essential+Perturb-seq+Gene+Perturbation+Signatures), PT:100 Ep, Tr:20 Ep, Dec:20 Ep. Params: JEPA 11M, AC 6.9M. Config: embd=256, pathways=1024, heads=4, layers=6.

The main updates were:

1. We improved our data handling to align with how GEARS benchmarking is done so we held out any cell with a perturbation that matched the training split on our dataset. This means that when we ran our benchmark the perturbation were completely unseen by our model.   
2. Pre-train the action-free model. This means we'll run masked training on the student/teacher model.  After that we freeze the student/teacher and then train the action predictor.  
3. We updated how perturbations were handled.  In V0.1 we simply gave each perturbation an integer.  For V0.2 we converted each perturbation to the amino acid sequence of the protein and then used ESM2 (esm2_t6_8M_UR50D) to create an embedding for each protein.
4. Removed RoPE since we do not care about position. 

## v0.1 Architecture

**Training:** [Gears K562](https://maayanlab.cloud/Harmonizome/dataset/Replogle+et+al.%2C+Cell%2C+2022+K562+Essential+Perturb-seq+Gene+Perturbation+Signatures) - CRISPRi perturbations of essential genes in K562 (CML) cells. Preprocessing: count normalization (1e4), log1p transform, top 4096 genes. (Removed due to data leakage.)

**Architecture:** Three-model system inspired by JEPA:
1. **Student** - Encodes control cell state to latent space
2. **Predictor** - Adjusts student embedding based on perturbation to predict perturbed state
3. **Teacher** - Encodes actual perturbed cell as prediction target (EMA-updated)

Encoders use Pre-Norm Transformer with RoPE. Predictor uses DiT-style blocks with AdaLN conditioning on the perturbation vector.

## Datasets (v0.6)

| **Dataset** | **Type** | **Cell Lines** | **Value Add** | **Status** |
| - | - | - | - | - |
| **Replogle K562 Essential** | CRISPRi | K562 (Leukemia) | **Baseline:** Deep coverage of cancer biology | v0.5+ |
| **Replogle K562 Genome-Wide** | CRISPRi | K562 (Leukemia) | **Scale:** Genome-wide perturbation coverage | v0.6 |
| **Replogle RPE1** | CRISPRi | RPE1 (Retinal) | **Generalization:** Non-cancer, normal karyotype | v0.6 |
| **Norman 2019** | CRISPRa (Dual) | K562 | **Physics:** Non-linear gene interactions ($A+B \neq A+B$) | v0.6 |
| **sciPlex (Srivatsan)** | Chemical | A549, MCF7, K562 | **Chemistry:** Drug-to-gene-state mapping (650k cells) | v0.6 |
| **Adamson 2016** | CRISPRi | K562 | **Stress:** High-resolution toxicity pathways | v0.6 |
