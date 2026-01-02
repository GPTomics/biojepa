import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import copy
import math
import numpy as np
import glob
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass

class BioMultiHeadAttention(nn.Module):
    # mirrors nn.MultiheadAttention(dim, heads, batch_first=True) 
    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.embed_dim % config.heads == 0
        
        self.head_dim = config.embed_dim // config.heads
        self.heads = config.heads
        
        # Projections
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x):
        B, T, C = x.size() # Batch, Seq, Embed Dim
        
        # 1. Project
        q = self.q_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # 2. Standard Scaled Dot Product Attention (Permutation Invariant)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        # 5. Reassemble
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.embed_dim, int(config.mlp_ratio * config.embed_dim))
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(int(config.mlp_ratio * config.embed_dim), config.embed_dim)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CellStateBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = BioMultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        # 1. Attention 
        x = x + self.attn(self.ln_1(x))

        # 2. MLP
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class CellStateEncoderConfig:
    num_genes: int = 4096
    num_pathways: int = 1024 
    n_layer: int = 24 
    heads: int = 12
    embed_dim: int = 768
    mlp_ratio: float = 4.0 # Changed to float for precision
    mask_matrix: np.ndarray = None 

class CellStateEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Learnable Network, initialized based on known pathwasy 1 == connection
        # wrapping as a "parameter" allows it to be learned
        assert config.mask_matrix is not None, 'Must provide binary_pathway_mask!'
        init_weights = torch.tensor(config.mask_matrix).float().T 
        self.pathway_weights = nn.Parameter(init_weights)
        
        # Learnable Gene Embeddings [num_genes, Dim]
        self.gene_embeddings = nn.Parameter(torch.randn(config.num_genes, config.embed_dim) * 0.02)
        
        # Context Injector
        self.total_count_proj = nn.Linear(1, config.embed_dim)

        # Transfomer
        self.blocks = nn.ModuleList([CellStateBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

        # Initiation 
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias)
        
    def forward(self, x_genes, x_total_ct):
        # 1. Project Genes
        x_genes = x_genes.unsqueeze(-1) 
        gene_repr = x_genes * self.gene_embeddings.unsqueeze(0)

        # 2. Gene Embeddings @ pathway weights
        x_pathway = self.pathway_weights @ gene_repr

        # 3. Context Injection
        x_total_ct = x_total_ct.unsqueeze(-1)
        x_total_ct = self.total_count_proj(x_total_ct)
        x_total_ct = x_total_ct.unsqueeze(1)
        x = x_pathway + x_total_ct

        # 4. Set Transformer
        for block in self.blocks:
            x = block(x)
        
        # 5. Layer Norm
        x = self.ln_f(x)

        return x

class AdaLN(nn.Module):
    '''
    Adaptive Layer Norm for conditioning the predictor on action embeddings.
    The action vector regresses the Scale (gamma) and Shift (beta) of the normalization.
    '''
    def __init__(self, embed_dim, action_embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.action_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(action_embed_dim, 2 * embed_dim)
        )
        # Initialize to identity (gamma=0, beta=0 originally, effectively gamma=1 after logic)
        # Zero-init the last layer so the action starts as a "no-op" (identity)
        nn.init.zeros_(self.action_mlp[1].weight)
        nn.init.zeros_(self.action_mlp[1].bias)

    def forward(self, x, action_emb):
        # x: [Batch, Seq, Dim]
        # action_emb: [Batch, action_embed_dim]
        
        # Project action to style: [B, 2*D] -> [B, 1, 2*D]
        style = self.action_mlp(action_emb).unsqueeze(1) 
        gamma, beta = style.chunk(2, dim=-1)
        
        # Apply affine transformation based on action
        return self.norm(x) * (1 + gamma) + beta

class PredictorBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Conditioning (AdaLN) replaces standard LayerNorm
        self.ada_ln1 = AdaLN(config.embed_dim, config.action_embed_dim)
        
        # 2. Attention (Using the shared BioMultiHeadAttention)
        self.attn = BioMultiHeadAttention(config)
        
        # 3. Conditioning (AdaLN) for the MLP block
        self.ada_ln2 = AdaLN(config.embed_dim, config.action_embed_dim)
        
        # 4. MLP (Using the shared MLP)
        self.mlp = MLP(config)

    def forward(self, x, action_emb):
        # 1. AdaLN -> Attention  -> Residual
        x_norm = self.ada_ln1(x, action_emb)
        x = x + self.attn(x_norm)
        
        # 2. AdaLN -> MLP -> Residual
        x_norm = self.ada_ln2(x, action_emb)
        x = x + self.mlp(x_norm)
        
        return x

@dataclass
class ACPredictorConfig:
    num_pathways: int = 1024
    n_layer: int = 6 
    heads: int = 4
    embed_dim: int = 384
    action_embed_dim: int = 256 
    mlp_ratio: float = 4.0
    max_perturb: int = 2058 ## eventually try to get to a 2**N power

class ACPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Action Embedding (Discrete ID -> Vector)
        self.action_embed = nn.Embedding(config.max_perturb, config.action_embed_dim)
        
        # Learnable Queries ("Mask Tokens") for the future state
        # One query vector per pathway position
        self.mask_queries = nn.Parameter(torch.randn(1, config.num_pathways, config.embed_dim) * 0.02)
        
        self.blocks = nn.ModuleList([
            PredictorBlock(config) for _ in range(config.n_layer)
        ])
        
        self.final_norm = AdaLN(config.embed_dim, config.action_embed_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, context_latents, action_ids):
        """
        context_latents: [Batch, N, Dim] (From Student Encoder)
        action_ids: [Batch] (Ints)
        """
        B, N, D = context_latents.shape
        
        # 1. Embed Action
        action_emb = self.action_embed(action_ids) # [B, action_embed_dim]
        
        # 2. Construct Input: [Context, Mask_Queries]
        # We concatenate the learned queries to the context. 
        # The predictor will attend to the context to update the queries.
        queries = self.mask_queries.repeat(B, 1, 1) # [B, N, D]
        sequence = torch.cat([context_latents, queries], dim=1) # [B, 2N, D]     
        
        # 3. Pass through AdaLN Blocks
        for block in self.blocks:
            sequence = block(sequence, action_emb)
            
        sequence = self.final_norm(sequence, action_emb)
        
        # 4. Return only the predicted part (The Queries corresponding to N..2N)
        predictions = sequence[:, N:, :] 
        return predictions

@dataclass
class BioJepaConfig:
    mask_matrix: np.ndarray
    num_genes: int = 4096
    num_pathways: int = 1024
    n_layer: int = 6 
    heads: int = 4
    embed_dim: int = 256
    action_embed_dim: int=256 
    mlp_ratio: float = 4.0
    max_perturb: int= 2058 ## eventually try to get to a 2**N power
    
class BioJepa(nn.Module):
    def __init__(self, config):
        super().__init__()

        enc_conf = CellStateEncoderConfig(
            num_genes=config.num_genes,
            num_pathways=config.num_pathways,
            n_layer=config.n_layer,
            heads=config.heads,
            embed_dim=config.embed_dim,
            mlp_ratio=config.mlp_ratio,
            mask_matrix=config.mask_matrix
        )
        
        self.student = CellStateEncoder(enc_conf)   
        self.teacher = copy.deepcopy(self.student)
        
        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False


        pred_conf = ACPredictorConfig(
            num_pathways=config.num_pathways,
            n_layer=config.n_layer,
            heads=config.heads,
            embed_dim=config.embed_dim,
            action_embed_dim=config.action_embed_dim,
            mlp_ratio=config.mlp_ratio,
            max_perturb=config.max_perturb
        )
        self.predictor = ACPredictor(pred_conf)

    def forward(self, x_control, total_control, x_case, total_case, action_id):
        # 1. Teacher
        with torch.no_grad():
            target_latents = self.teacher(x_case, total_case)
            
        # 2. Student 
        context_latents = self.student(x_control, total_control)
        
        # 3. Predictor 
        predicted_latents = self.predictor(context_latents, action_id)
        
        # 4. Latent Loss (L1)
        loss = F.l1_loss(predicted_latents, target_latents)
        
        return loss

    @torch.no_grad()
    def update_teacher(self, m=0.996):
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.mul_(m).add_((1 - m) * param_s.data)
            