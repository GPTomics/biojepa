**In progress version:**  v0.7
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

*v0.6 results are preliminary and under investigation. See notes below each section.

### Pretraining Evals (encoder quality)

v0.6 trains on 6 datasets (10K genes) vs v0.5 single dataset (5K genes). Metrics not directly comparable due to task difficulty differences. Full test: 309,760 samples, 1,085 perturbations, 5 datasets.

| Metric | v0.6* | v0.5 | v0.4 | Context |
| - | - | - | - | - |
| Batch Invariance Ratio | **0.851** | 0.167 | 0.215 | Higher = more bio vs technical. Within-dataset macro mean: 0.784 |
| Perturbation Detection AUROC | **0.566** | — | — | norman 0.700, k562e 0.693, adamson 0.619, k562gw 0.550, sciplex 0.544 |
| Reconstruction Pearson | **0.986** | — | — | Gene expression from embeddings |
| Cell Type Accuracy | **99.3%** | — | — | 3 cell types in test data |
| Embedding Consistency | **0.848** | — | — | Same-pert similarity ratio |
| Essential Gene AUROC | 0.630 | **0.741** | 0.707 | v0.5 had K562-only home-field advantage |
| KEGG Silhouette | **-0.072** | -0.083 | — | Pathway structure in gene embeddings |
| Effective Dimensionality (90%) | **45** | — | — | Of 256 total dims |

### Alignment Evals (composer quality)

New in v0.6. Evaluates dual-pathway ActionComposer (sequence + target fusion).

| Metric | v0.6* | Context |
| - | - | - |
| Mode Sensitivity Accuracy | **55.2%** | 7 modes, chance = 14.3% (3.9x) |
| Paired Cosine Similarity (DNA) | 0.121 | Seq-target alignment quality |
| Seq-to-Target Retrieval MRR | 0.0013 | Target: >0.6. Alignment weak |
| Cross-Modality Consistency | **3.063** | Target: >>1. Same-target perts cluster |
| Fused-to-Seq Cosine | 0.865 | Fusion still seq-dominated |
| Target Family Probing (target) | **23.6%** (66x) | Raw ESM-2 embeddings carry signal |
| Target Family Probing (fused) | **17.8%** (50x) | Fusion leverages some target info |

### Full Model Evals (expression prediction)

v0.6: 1,085 test perturbations, 10K genes, 5 datasets (309,760 samples). v0.5: 286 test perturbations, 5K genes, 1 dataset.

| Metric | v0.6* | v0.5 | v0.4 | v0.3 | v0.2 | Context |
| - | - | - | - | - | - | - |
| **R² Top 50 DEGs (mean)** | **0.816** | 0.066 | 0.096 | 0.060 | -0.027 | The hard test |
| **R² Top 50 DEGs (median)** | **0.849** | 0.341 | 0.325 | 0.255 | 0.269 | The hard test |
| **Severity Spearman** | **0.638** | 0.471 | — | — | — | Predicting effect magnitude |
| Severity Pearson | 0.738 | 0.835 | **0.870** | — | — | Predicting effect magnitude |
| Global MSE | **0.204** | 0.489 | 0.498 | 0.515 | 0.790 | Lower is better |
| Pearson R (Top 20) | 0.869 | 0.919 | **0.927** | 0.921 | 0.605 | Harder task (more perts/genes) |
| R² All Genes (mean/median) | **0.964 / 0.971** | 0.930 / 0.940 | 0.918 / 0.927 | 0.902 / 0.910 | 0.942 / 0.956 | Inflated (most genes don't change) |
| Direction Accuracy (All) | **98.9%** | 89.7% | 87.7% | — | — | UP/DOWN/UNCHANGED |
| Direction Accuracy (Top 50) | **77.6%** | 28.5% | 34.0% | — | — | On genes that change most |
| Centroid Accuracy | **0.034** | — | — | — | — | New hard metric (random ~0.001) |
| Pearson Delta (all genes) | **0.216** | — | — | — | — | New hard metric, baseline-adjusted |
| vs Baseline Beat Rate | 0.9% | — | — | — | — | New hard metric |
| DEG Precision@20 | 1.9% | **4.8%** | 3.1% | — | — | Random baseline ~0.1% |
| DEG vs Random @20 | 9.6x | **12.0x** | 7.8x | — | — | Improvement over chance |
| MOA Similarity Ratio | **1.142** | 1.005 | 1.006 | — | — | >1 = same-pathway more similar |
| MOA p-value | **6.4e-99** | 0.267 | — | — | — | Statistical significance |
| Combo Pearson Delta | 0.286 | — | — | — | — | 8 Norman dual-gene perturbations |
| Combo Additive Baseline | 0.781 | — | — | — | — | Additive model outperforms |
| Retrieval MRR (DNA) | 0.0005 | **0.007** | 0.010 | — | — | Bank: 11.6K vs 1.25K |
| Retrieval MRR (Chemical) | **0.036** | — | — | — | — | New capability (188 compounds) |
| Uncertainty ECE | 0.572 | 0.281 | **0.135** | — | — | Lower is better. v0.6 anti-calibrated |
| Action Vector Silhouette | **-0.339** | -0.412 | — | — | — | Pathway structure in action vectors |

#### Dataset Breakdown

| Dataset | Perts | Pearson Delta | R² Top-50 DEGs | Centroid Acc |
| - | - | - | - | - |
| k562gw | 1,053 | 0.232 | 0.828 | 0.068 |
| k562e_raw | 286 | 0.221 | 0.783 | 0.017 |
| adamson | 9 | 0.018 | 0.832 | 0.222 |
| norman | 8 | 0.286 | 0.679 | 0.125 |
| sciplex | 14 | 0.078 | 0.956 | 0.143 |

#### GEARS Benchmark

| Dataset | Perts | Splits | Pearson All Genes | Pearson Delta |
| - | - | - | - | - |
| Replogle K562 | 286 | GEARS official | 0.976 | 0.221 |
| Adamson | 9 | NOT GEARS | 0.983 | 0.018 |

**Notes:** R² on all genes looks good because ~95% of genes barely change -- R² on Top 50 DEGs is the real test. Centroid accuracy and vs-baseline beat rate are new hard metrics: centroid accuracy measures whether each perturbation's predicted centroid is closest to the correct actual centroid (random ~1/1085), and beat rate measures how often the model outperforms a mean baseline. The R² top-50 DEGs jump (was 0.442 on partial test, now 0.816 on full test) needs investigation -- may partly reflect full-test composition (k562gw dominates with 1,053 of 1,085 perts). Adamson pearson_delta ~0 suggests model is not learning perturbation-specific effects on that dataset. Combo perturbations (Norman dual-gene) lose to an additive baseline (model: 0.286 vs additive: 0.781 pearson delta). For detailed metric interpretation, see `docs/eval_planning.md`. For SOTA analysis, see `biojepa_private/docs/sota_evals.md`.

### What Makes BioJEPA Different

| Capability | BioJEPA | GEARS | scLAMBDA | LPM |
|------------|---------|-------|----------|-----|
| Genetic perturbations | Yes | Yes | Yes | Yes |
| Chemical perturbations | Yes | No | No | Yes |
| Multi-pert (>1 simultaneous) | Up to 4 | 2 (graph) | No | No |
| Uncertainty quantification | Yes (Gaussian NLL) | No | No | No |
| Missing data fallback | Yes | N/A | N/A | Partial |
| Mode conditioning | 9 modes (FiLM) | No | No | No |
| Dose conditioning | Yes (multiplicative) | No | No | No |

## v0.7 Changes

1. **Loss Component Logging:** TensorBoard logging in all four training stages. `vicreg_loss` and forward methods accept `return_components` to return loss breakdowns (NLL, std, cov). Default log directory: `checkpoint_dir/training_logs/{stage}/`.
    1. Benefit: Enables real-time monitoring of individual loss terms during training, making it possible to diagnose training dynamics (e.g. VICReg dominating NLL, or variance collapse) without post-hoc analysis.
2. **Renamed Training Stages:** Pretraining/Alignment/Full Training renamed to Encoder Training/Composer Training/AC Training. Updated across config classes, training functions, eval entry points, CLI modules, checkpoint prefixes, and notebooks. `VERSION` constant in `config_v0_7.py` controls checkpoint filenames.
    1. Benefit: Names now describe what each stage trains rather than using generic terms, reducing confusion when discussing the four-stage pipeline.
3. **Conditioning Dropout for AC Training:** Added a learnable `null_action` parameter and `p_uncond` probability to `BioJepa.forward()`. During training, with probability `p_uncond` the entire batch's action latents are replaced with the learned null vector, forcing the predictor to make predictions without perturbation information.
    1. Benefit: Prevents the ACPredictor from ignoring the action conditioning signal. Without this, the predictor can minimize loss by learning cell-state-only patterns (producing a 0.9% beat rate). Conditioning dropout forces the model to rely on context when action is absent, and to actually use the action signal when present. Validated by dbDiffusion (PNAS 2025) and scLDM (2025) for perturbation-conditioned prediction.
4. **Full-Position AC Loss + Mask Annealing to Zero:** NLL loss is now computed on all gene positions rather than only masked positions. Added `mask_anneal_floor` config field (default 0.0) replacing the hardcoded floor of 0.1, enabling clean annealing all the way to mask_ratio=0.0.
    1. Benefit: Bridges the train/eval distribution gap. Training previously masked 76.6% of genes but inference uses no masking, so the predictor never saw full-context encoder output. Full-position loss decouples masking (an encoder input concern) from supervision (a predictor concern), so reducing the mask ratio no longer reduces the amount of loss signal.
5. **Unknown Gene Handling + Context L2 Supervision:** Added a learnable `unknown_token` to `CellStateEncoder` for genes not measured by a dataset (~15-18% of positions in dominant datasets). Encoder training now uses a three-way loss partition: masked+measured positions get L1 loss (predict), unmasked+measured positions get MSE loss with a ramped coefficient (preserve context quality), and unmeasured positions get no loss (don't hallucinate from zero-filled inputs). Gene masks are loaded from shards and threaded through all training stages and evals.
    1. Benefit: Removes corrupted positions from loss computation. Previously, zero-filled unmeasured genes were indistinguishable from genuinely unexpressed genes, wasting model capacity and polluting loss terms. The context L2 term (inspired by V-JEPA 2.1's dense predictive loss) adds meaningful supervision to unmasked positions, ensuring the masked predictor preserves context representations rather than corrupting them through attention with neighboring mask tokens.
6. **WSD LR Schedule + EMA Momentum Annealing:** Replaced OneCycleLR with a Warmup-Stable-Decay (WSD) schedule: linear warmup, constant LR through the stable phase, then linear decay to near-zero. During the decay phase (Phase 2), EMA momentum optionally anneals linearly toward 1.0 (freezing the teacher). The context L2 coefficient ramps from 0 to its target value independently.
    1. Benefit: Constant LR during the stable phase keeps the model at full learning capacity longer (OneCycleLR was at ~10% by epoch 80). The Phase 1 checkpoint is reusable as a branch point for multiple Phase 2 experiments. EMA annealing prevents late-training loss spikes caused by teacher drift when the student's LR is near-zero. Pioneered by MiniCPM, validated by VL-JEPA's constant-LR pretraining.
7. **Modality-Balanced Sampling for Composer Training:** Chemical perturbations are only 1.5% of alignment training data (166/10,797 samples). At batch_size=64, most batches contain zero or one chemical sample, starving the chemical projector and inhibitor FiLM of training signal. A new `chemical_fraction` config field on `ComposerTrainingConfig` controls row duplication in `ComposerLoader`: chemical samples are duplicated to reach the target fraction (e.g., 0.1 for ~10%), then normal permutation distributes them across batches. Only applies to the train split.
    1. Benefit: Ensures the chemical pathway gets meaningful gradient signal every batch (~6 chemical samples per batch at 10% target), improving chemical MRR and inhibitor/agonist/degrader mode conditioning without changing the loss function or model architecture.
8. **SigLIP Sigmoid Loss for Composer Training:** Composer alignment uses SigLIP (Sigmoid Loss for Language-Image Pre-training) pairwise sigmoid loss: each (seq, target) pair contributes gradients independently, removing the softmax bottleneck where only 63 negatives exist at batch_size=64. A learnable bias parameter (`siglip_bias`, initialized at -10.0 per the SigLIP paper) sets the decision boundary. Temperature remains fixed at 0.012 from HPO.
    1. Benefit: At small batch sizes, softmax normalization dilutes gradient signal across few negatives. SigLIP treats each of the B^2 pairs independently, providing stronger per-pair gradients. The SigLIP paper shows sigmoid loss significantly outperforms softmax at batch sizes below 16K. The learned bias naturally handles the positive/negative imbalance (64 positives vs 4,032 negatives at BS=64).
9. **Dose Conditioning for AC Training:** Added a `dose_proj` MLP to ActionComposer that applies multiplicative dose scaling to each perturbation slot's action latent after mode conditioning. The dataloader applies `log1p` to raw dose values (compressing the 100x range of 0.1-10.0 micromolar to ~0.095-2.398) while preserving the -1.0 sentinel for genetic perturbations. Identity initialization (zero weights, ones bias on the final layer) ensures existing checkpoint compatibility via `strict=False` loading. The `dose=None` default on all modified signatures means non-dose paths have zero overhead.
    1. Benefit: Previously, sci-Plex samples with the same drug at different doses produced identical action latents, creating contradictory training signals (same conditioning, different outcomes). Dose conditioning resolves this ambiguity and enables the model to learn dose-response relationships, improving chemical perturbation prediction and dose-response eval metrics (v0.6: monotonicity ~50%, Spearman -0.151, both essentially random).

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
8. **Latent Masking for Action Predictor:** Added masking on our action predictor so that a portion of (`n=0.15`) of the input vector is masked during training of the ACPredictor.
    1. Benefit: Pushes the model to better learn the relationship between perturbations and cell states, improving prediction accuracy over implicit positional learning.
9. **Beta-NLL Loss (beta=0.5):** Replaced standard Gaussian NLL with a beta-weighted variant that scales each sample's loss by `variance.detach().sqrt()`. The detached variance breaks the gradient coupling that allows the model to reduce loss by inflating its uncertainty estimate rather than improving its predictions.
    1. Benefit: Fixes a known pathology in heteroscedastic regression where the model learns to predict high variance everywhere to trivially minimize NLL. With beta=0.5, the model must actually improve its mean predictions to reduce loss.
10. **Mask Annealing:** Optionally reduces the gene mask ratio during the final fraction of full training epochs, linearly interpolating from the base mask ratio down to a floor of 0.1. Controlled by `mask_anneal_pct` (default 0.0 = disabled).
    1. Benefit: Bridges the train-test mismatch where training uses partial gene masking but inference sees all genes. Gradually exposing the model to less masking in late training improves generalization to the unmasked inference setting.


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
