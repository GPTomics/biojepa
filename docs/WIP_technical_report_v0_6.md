# CURRENTLY A WIP!!



# BioJEPA-AC v0.6 - A world model for cells

## Evals

### Action Conditioned Predictor

BioJEPA-AC primary benefit is not creating the joint-embedding space, but being able to take actions, in our case perturbations, and move cell representations in that embedding space.   To evaluate if our latent space and understanding of moving representations across it are useful, we use a series of evals both directly on the latent space, and with lightweight decoder heads.  

![full_model_overview](../resources/v0_6/ac_eval.png)

*Fig X. Forward pass of our action conditioned network*

To do the evals, we take a control cell state, add up to four perturbation representations, and then do a forward pass on our BioJEPA-AC model.  This generates a representation of the perturbed cell state, after which we pass that to the task eval.   We have 8 different evals that we do on our fully trained model. 

### Eval Heads

The BioJEPA-AC model is an encoder meaning its output is an embedding representation, not a specific molecular representation or property.  While this is what allows us to do energy-based learning, it makes it difficult to know how bilogically valuable the model is as a foundation model, something done with our evals.  To complete the evals we need to add a decoder head on top, but, we need to ensure the evals are not so intelligent/capable that they mask issues in our base model.  To avoid this, our heads use a single learnable liear layer to project our latent representations to the appropriate eval dimensions.  This results in 2 different heads: a linear expression decoder and a linear classifier.

![eval decoder overview](../resources/v0_6/eval_decoder_overview.png)

*Fig X. An overview of the different heads, an expression predictor and a general linear classifier, we use to conduct evaluation benchmarking on our BioJEPA-AC model*

#### Linear Expression Decoder

Since our cell state learning data is perturbSeq, we have a lot of gene expression information in our model. Because of this, one of our main benchmarks will be to see how well we can predict perturbation based changes in expression. For our model to do this, we'll have to project from our latent space down to a single value per gene.  Since our cell state latent representation is already $[\text{n\_genes}, \text{embed\_dim}]$, this decoder will use its single linear layer to convert its latent representation to a single value per gene. Recall that our model generates both a mean and logvar representation but you'll see that our linear expression prediction only takes in a single input. As you'll see in the eval deep dive, we use the mean for our expression prediction and reserve the logvar for error analysis. 

#### General Linear Classifier

Another common eval we perform both on our cell latent, and on our intermediate modules is predicting different input metadata and classes (e.g. cell batch information, cell types, if a cell is perturbed, perturbation modes, perturbation pathways).  Both our BioJEPA-AC model and the components all are encoders so we have to build a classifier that can predict the class based on the latent representation.  Given we use it in multiple evals, we built a general purpose classifier class.  This class takes in the latent representation and, with a single linear later, projects it to the class dimension. This linear calcualtion createa a value per class that represents its probabilities.  In some cases like classifier a cell latent to a cell type, we do an additional step of mean pooling. We do this since the classifier will generate a class prediction per gene and we need to collapse it down to per cell without givint he model more intelligence. As you'll see in the eval deep dives, the classifier leaves the values in their raw state since our deifferent evals will subprocess the data in different manners.  

### Eval Data Splits - Hold Out 

To create our hold-out datasets, we followed an 85% / 5%/ 10% split between Train, Val, Test.  Before we applied this logic, we first used the [GEARS python package](https://github.com/snap-stanford/GEARS/tree/master) to identify the heldout Train, Val, Test split for Replogle K562-essential. This dataset uses a 67.5%,  7.5%, 25.0% train, val test split that we started with.  The split identifies unique perturbation to hold out.  With this as the start, we built a hold-out perturbation list that ensured we hit the 15% held out pertubations on each dataset and, that those perturbations would be held out across all datasets.  This methodology ensures that we do not have perturbation leakage.  Additionally this allows us to do a direct head-to-head comparison for the Replogle K562-essential dataset based on what we held out.  

For the reamining datasets, since we exptracted perturbations at random, we will have to compare against published numbers. If we have time we'll attempt to run the models on the same splits we have (if we can fennagle the data in).  

### Full Model Evals

#### Expression Prediction

The main goal of this eval is to see if we can predit post perturbation gene expression.  This eval is useful as a high accuracy in gene expression can lead to an explainable understanding on perturbation effects for viability and gene network impact. In v0.6 we're focusing on 10,000 genes. Since we're covering almost half of the known genes, many of the genes will not have signficant changes. This means that we have to be careful to not evaluate just the whole gene set as predicting no change can be highly accurate for most genes.  To avoid falling into this trap, we run a number of different analyses including focusing on the top 20 and top 50 differentially expressed genes allowing us to see if the model is able to predict the largest movement vs just the housekeeping genes.

While we've done a number of different analysis, we'll dive into the following commonly performed evals. A major component of our expression benchmark is not looking at raw prediction made by the linear expression decoder, but the relative prediction to the the predicted control. Our baseline *real delta* is just the observed change in expression in the raw data as calculated by $\delta_g = x^{\text{case}}_g - x^{\text{ctrl}}_g$.  To calcluated our *predicted delta*, we first pass the latent representation of our control cell through the linear expression decoder and then subtract the pertubed expression prediction from it, calculating $\hat{\delta}_g = \hat{x}^{\text{case}}_g - \hat{x}^{\text{ctrl}}_g$. We use the predicted control expression to isolate our expression prediction into terms of BioJEPA-AC's learned latent space, ensuring that if it learns a different baseline for the control but knows the distance to move the case cell, it can still be correct.  The expression deltas only work for some of our evals while others rely on absolute prediction. To create our absolutes, we look at the raw data for our *real absolute* expression. For our *predicted absolute* we add our predicted delta to our real control, calcluating $\hat{x}^{\text{abs}}_g = x^{\text{ctrl}}_g + \hat{\delta}_g$.    These four values, along with our control form the basis of our evaluation benchmark. 

For each of the following benchmarks, I'll explan how the calcluation is done, and then report our the performance per dataset.  I'll add commentary where appropriate. 

##### Sample Level Pearson's Correlation Coefficient For Top 20 DEGs

For each sample we calculate the Pearson's correlation coefficient, $r$, on the top 20 differentially expressed genes (DEGs). Pearson's correlation coefficient is a measure of how linearly correlated the predicted and real expression profiles are.  For our calculation, we focus on evaluating the change in gene expression, or *delta*. The value of this eval is that if the predicted expression has a high correlation with the real expression, but the values are off, our model is still useful as it's learned the profile.  Since our dataset has 10,000 genes, many will have minor adjustments to expression; therefore, we only evaluate the genes where we see the largest changes in expression. The genes are identified as the ones with the highest *real delta* absolute values. We then compare those against the *predicted delta* using: 
$$
r_{\text{sample}} = \frac{1}{N}\sum_{i=1}^{N}\frac{\sum_{k=1}^{K}(\hat{\delta}_{i,k} - \bar{\hat{\delta}}_i)(\delta_{i,k} - \bar{\delta}_i)}{\sqrt{\sum_{k=1}^{K}(\hat{\delta}_{i,k} - \bar{\hat{\delta}}_i)^{2}} \cdot \sqrt{\sum_{k=1}^{K}(\delta_{i,k} - \bar{\delta}_i)^{2}}}
$$

where $K=20$ genes are selected per sample as the largest $|\delta_{i,g}|$. 

We see the folloiwng performance by dataset

|Dataset|BioJEPA-AC v0.6|Dataset X|
|-|-|-|
| Adamson|0.8330||
| Replogle K562 essential|0.8473||
| Replogle K562 genome-wide|0.8900||
| Norman|0.7634||
| Sciplex|0.8411||

##### Perturbation Level Mean Pearson Correlation Coefficient







pearson_all_genes, pearson_delta_all_genes

|Dataset|BioJEPA-AC v0.6|Dataset X|
|-|-|-|
| Adamson|||
| Replogle K562 essential|||
| Replogle K562 genome-wide|||
| Norman|||
| Sciplex|||

##### $R^2$ Top 50 DEG 
|Dataset|BioJEPA-AC v0.6|Dataset X|
|-|-|-|
| Adamson|||
| Replogle K562 essential|||
| Replogle K562 genome-wide|||
| Norman|||
| Sciplex|||

##### Cross Perturbation Centroid Accuracy

centroid_accuracy
|Dataset|BioJEPA-AC v0.6|Dataset X|
|-|-|-|
| Adamson|||
| Replogle K562 essential|||
| Replogle K562 genome-wide|||
| Norman|||
| Sciplex|||

##### Control Beat Rate

vs_baseline.beat_rate
|Dataset|BioJEPA-AC v0.6|Dataset X|
|-|-|-|
| Adamson|||
| Replogle K562 essential|||
| Replogle K562 genome-wide|||
| Norman|||
| Sciplex|||













 For a detailed breakdown of the eval see our [explainer notebook](https://github.com/GPTomics/biojepa/blob/main/layer_explainers/explainer_eval_expr_prediction.ipynb).

#### Gene Level Analysis

#### Perturbation Retrieval

#### Uncertainty Calibration

#### Action Vector Pathway

#### Mechanism of Action Matching

#### Combination Perturbation Impact

#### Dose Response









## Abstract

This report presents an updated view on BioJEPA-AC (Biological Joint-Embedding Predictive Architecture - Action Conditioned), a joint-embedding predictive architecture that uses a shared latent space to learn cell dynamics and perturbation response.

## Model Architecture

### Overview

![full_model_overview](../resources/v0_6/full_model_overview.png)

*Fig X. BioJEPA-AC v0.6 inference overview highlighting how a cell state and a perturbation are converted into a mean and variance latents representing the cell state*

The v0.6 BioJEPA-AC architecture is designed to create predict a latent representation of cell state given it's baseline cell state and up to 4 different perturbatios.  The model is designed to handle cell states represented across 10,000 genes with continuous value expression counts that may have extreme sparsity inherent to Perturb-seq data.  Through the use industry foundation models and the action composer, the model can handle up to 4 combined predefined or novel perturbations across different modalities. The architecture comprises of the following modules: 

* **Cell State Encoder**:  This transformer based module maps the latent space representing cell states. It builds the latent space based on the relative gene expression and total cell expression. It is used for both the *Context Encoder* and *Target Encoder*.
* **Action Composer**:  This module is a dual-pathway encoder with additive fusion and FiLM-based mode conditioning.  It creates a unified latent space for the embedding representations of the different types of perturbations.
* **Action Conditioned Predictor**: This transformer based module uses the action latents to adjust the cell state representation in the latent space to generate the latent representation of the perturbed cell state.

### Cell State Encoder

![context encoder overivew](../resources/v0_6/encoder_overview.png)

*Fig X. The data that creates our cell state representation in the latent space*

The cell state encoder architecture is our foundation fo the joint-embedding pretrained architecture. This architecture is both used for our context encoder (student) and target encoder (teacher).  The encoder is a transformer-based with linear attention, SwiGLU and RMSNorm. We use both the count/log normalized expression counts and the sum of total expression to build out our cell latent.  We include the total expression sum to ensure our models have a way to flag unviable cells that would look like just "noisy" expression if we were to only look at the normalized per gene expression.  Our cell representation ends with a representation in the $[\text{token},\text{embedding}]$  space.  Currently the tokens for the cell are the different genes that we have expression infromation for, so in  the case of v0.6, the 10,000 highest expressed genes across our datasets.  The architecture though is built that we can expand to non-gene based cell representaed tokens. Review our [explainer](https://github.com/GPTomics/biojepa/blob/main/layer_explainers/explainer_cell_state_encoder.ipynb) notebook for a deeper dive.

### Action Composer

![action composer overview](../resources/v0_6/action_composer_overview.png)

*Fig X. An overview of how we merge different types of perturbations with different data into the same perturbation representation.*

The action composer is a dual-pathway linear encoder that projects perturbation embeddings into a shared latent space via separate linear projections, fuses them, and applies FiLM conditioning from a learned mode embedding to encode the perturbation with mechanism aweareness. This architecture allows us to pull our different perturbations into the same unified perturbation space and can then feed it into our action-conditioned predictior. Our perturbation representation ends with a representation in the $[\text{n\_perts},\text{pert\_embedding}]$  space.

This architecture is designed to suppor our goal of cells with multiple perturbations (up to 4 for v0.6) that may be different modalities and modes.  Perturbations can be targeted genetic (e.g. CRISPR) perturbations or therpeutics ranging from nucleic acids, proteins, and small molecules. We require a perturbations has either a sequence and/or target (at least one), and the mode (crispri, crispra, overexpression, knockout, inhibitor, agonist, degrader, binder, unknown).  To improve our flexibility and generalization of our action composer, instead of taking in the raw representation of the sequence and target, during data prep we pass them through bio foundation models to get embedding representations. We rely on the action composer understanding how to create embeddings to represent functionally similar, sequence different perturbations. By leveraging the BioFMs, we can easily expand to unseen and completely novel perturbations. Review our [explainer](https://github.com/GPTomics/biojepa/blob/main/layer_explainers/explainer_action_composer.ipynb) notebook for a deeper dive.

### Action Conditioned Predictor

![Action Conditioned Predictor Overview](../resources/v0_6/ac_predictor_overview.png)

*Fig X. An overview of how we shift the cell state in the shared latent based on the pertrubation*

The action conditioned predictor architecture is the foundation of the "AC" portion of our BioJEPA-AC model.  The predictor is transformer-based with smilar linear attention, SwiGLU, and RMSNorm.  It differs from the cell state encoder since it employs two different attention layers: cross attention to shift the cell state based on the pertrubation, and self attention to allow the cell state to learn from itself. The predictor uses a target indices to allow for predicting only a portion of the cell state representation, similar to how masked models operate.  It fuses the un-perturbed cell representation generated by the context encoder with the targets, and then uses attention to layer in the action latent generated from the action composer.  The predictor generates a predicted representation of the perturbed cell state including an uncertainty estimate, each of which is represented in the  $[\text{token},\text{embedding}]$  space. The idea is that this is the shared latent space that the cell state encoder learns and the predictor is just learning where to move the representation based on all of the perturbations. 

Review our [explainer](https://github.com/GPTomics/biojepa/blob/main/layer_explainers/explainer_ac_predictor.ipynb) notebook for a deeper dive.

---



# Evals

### Encoder Training Evals

        'batch_invariance': _batch_invariance(ctx),
        'gene_embedding_pathways': _gene_embedding_pathways(ctx),
        'essential_gene_prediction': _essential_gene_prediction(ctx),
        'cell_type_probing': _cell_type_probing(ctx),
        'reconstruction': _reconstruction(ctx),
        'perturbation_detection': _perturbation_detection(ctx),
        'embedding_consistency': _embedding_consistency(ctx),
        'latent_space_health': _latent_space_health(ctx),

**Batch invariance**

* batch classifier - accuracy, chance, above chance ratio
* perturb classifier - accuracy, chance, above chance ratio

**Gene Embedding Pathways**

* Silhouette score, knn accuracy, knn std, n classes, n samples
* Datasets: Kegg, Reactome

**Essential Gene Prediction**

* regression: pearsons, spearman
* classification: auroc, n_essentia_test, n_non_essential_test

**Cell Type Probing**

* Accuracy, Macro F1, Chance, Above chance ratio

**Reconstruction**

* MSE, Pearson R, Pearson R^2 

**Perturbation Detection**

* Aurora, accuracy, chance

**Embedding Consistency**

-  intra distance, inter distance, inter intra ratio, 

**Latent Space Health**

- effective dimensionality, variance, isotropy



### Copmposer Training Evals

### AC Training Evals

# Training



# Data Prep



# Bayesian Optimization

