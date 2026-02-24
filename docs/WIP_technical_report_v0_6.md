# CURRENTLY A WIP!!





# BioJEPA-AC v0.6 - A world model for cells

## Abstract

This report presents an updated view on BioJEPA-AC (Biological Joint-Embedding Predictive Architecture - Action Conditioned), a joint-embedding predictive architecture that uses a shared latent space to learn cell dynamics and perturbation response.

## Model Architecture

### Overview

![full_model_overview](/Users/djemec/code/biojepa_unified/biojepa/resources/v0_6/full_model_overview.png)

*Fig 1. BioJEPA-AC v0.6 inference overview highlighting how a cell state and a perturbation are converted into a mean and variance latents representing the cell state*

The v0.6 BioJEPA-AC architecture is designed to create predict a latent representation of cell state given it's baseline cell state and up to 4 different perturbatios.  The model is designed to handle cell states represented across 10,000 genes with continuous value expression counts that may have extreme sparsity inherent to Perturb-seq data.  Through the use industry foundation models and the action composer, the model can handle up to 4 combined predefined or novel perturbations across different modalities. The architecture comprises of the following modules: 

* **Cell State Encoder**:  This transformer based module maps the latent space representing cell states. It builds the latent space based on the relative gene expression and total cell expression. It is used for both the *Context Encoder* and *Target Encoder*.
* **Action Composer**:  This module is a dual-pathway encoder with additive fusion and FiLM-based mode conditioning.  It creates a unified latent space for the embedding representations of the different types of perturbations.
* **Action Conditioned Predictor**: This transformer based module uses the action latents to adjust the cell state representation in the latent space to generate the latent representation of the perturbed cell state.

### Context Encoder

### Action Composer

### Action Conditioned Predictor







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

