import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from dataclasses import dataclass

torch.manual_seed(1337)


# utils
def init_weights_robust(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        if isinstance(module, nn.Embedding):
            fan_in = module.embedding_dim
        else:
            fan_in = module.weight.size(1)
        std = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.02
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# modules 
class BioLinearAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.embed_dim % config.heads == 0
        
        self.head_dim = config.embed_dim // config.heads
        self.heads = config.heads
        
        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x, kv=None):

        B, T_q, C = x.size()
        kv_input = kv if kv is not None else x
        T_kv = kv_input.size(1)
        
        # 1. Project
        q = self.q_proj(x).view(B, T_q, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, T_kv, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, T_kv, self.heads, self.head_dim).transpose(1, 2)
        
        # 2. Apply Feature Map (ELU + 1)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        # 3. Linear Attention Calculation: Q @ (K.T @ V)
        # Aggregate global context from K and V
        kv_matmul = k.transpose(-2, -1) @ v   # [B, Heads, Head_Dim, Head_Dim]
        
        # Normalization term (denominator)
        k_sum = k.sum(dim=-2).unsqueeze(-1)   # [B, Heads, Head_Dim, 1]
        z = 1.0 / (q @ k_sum + 1e-6)

        # Compute Output (numerator * denominator)
        y = (q @ kv_matmul) * z
        
        # 4. Reassemble
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)
        y = self.c_proj(y)
        
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
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
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = BioLinearAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class PredictorBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.action_attn = BioLinearAttention(config)

        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.self_attn = BioLinearAttention(config)

        self.ln_3 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x, action_emb):

        # 1. Mechanism Injection (Cross-Attention)
        x_norm = self.ln_1(x)
        x = x + self.action_attn(x_norm, kv=action_emb)

        # 2. Dynamics Propagation (Self-Attention)
        x_norm = self.ln_2(x)
        x = x + self.self_attn(x_norm)

        # 3. Processing
        x_norm = self.ln_3(x)
        x = x + self.mlp(x_norm)
        
        return x

class MaskedPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Shallow transformer for reconstruction (typically fewer layers than encoder)
        self.blocks = nn.ModuleList([
            CellStateBlock(config) for _ in range(config.n_layer) 
        ])

        self.norm = nn.LayerNorm(config.embed_dim)
        self.pred_head = nn.Linear(config.embed_dim, config.embed_dim)

        self.apply(init_weights_robust)
                    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.pred_head(x)
        return x

@dataclass
class CellStateEncoderConfig:
    num_genes: int = 8192
    n_layer: int = 24 
    heads: int = 12
    embed_dim: int = 768
    mlp_ratio: float = 4.0 

class CellStateEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Learnable Gene Embeddings [num_genes, Dim]
        self.gene_embeddings = nn.Parameter(torch.randn(config.num_genes, config.embed_dim) * 0.02)
        
        # Total Count Injector
        self.total_count_proj = nn.Linear(1, config.embed_dim)

        # Transfomer
        self.blocks = nn.ModuleList([CellStateBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

        # Initiation 
        self.apply(init_weights_robust)
        
    def forward(self, x_values, total_counts):
        # 1. Project Genes
        x = x_values.unsqueeze(-1) 
        x = x * self.gene_embeddings.unsqueeze(0)

        # 2. Total Count Injection
        x_total_ct = total_counts.unsqueeze(-1)
        x_total_ct = self.total_count_proj(x_total_ct)
        x_total_ct = x_total_ct.unsqueeze(1)
        x = x + x_total_ct

        # 4. Set Transformer
        for block in self.blocks:
            x = block(x)
        
        # 5. Layer Norm
        x = self.ln_f(x)

        return x

@dataclass
class ACPredictorConfig:
    num_genes: int = 8192
    n_layer: int = 6 
    heads: int = 4
    embed_dim: int = 384
    mlp_ratio: float = 4.0
    pert_embd_dim: int = 320 

class ACPredictor(nn.Module):
    def __init__(self, config, pert_embd):
        super().__init__()
        self.config = config
        
        # Action Embedding (Discrete ID -> Vector)
        self.register_buffer('pert_bank', torch.tensor(pert_embd, dtype=torch.float32))

        # Perturbation Embedding
        self.adapter = nn.Sequential(
            nn.Linear(config.pert_embd_dim, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim)
        )
        
        # Learnable Queries for all tokens (genes)
        self.mask_queries = nn.Embedding(config.num_genes, config.embed_dim)
        
        self.blocks = nn.ModuleList([
            PredictorBlock(config) for _ in range(config.n_layer)
            ])
        
        self.final_norm = nn.LayerNorm(config.embed_dim)

        # Stochastic Heads (Mean & LogVar)
        self.head_mu = nn.Linear(config.embed_dim, config.embed_dim)
        self.head_logvar = nn.Linear(config.embed_dim, config.embed_dim)
        
        # initiation
        self.apply(init_weights_robust)

    def forward(self, context_latents, action_ids, target_indices):
        B, C_Len, D = context_latents.shape

        # 1. Prepare Action [Batch, 1, Dim]
        raw_pert = self.pert_bank[action_ids]
        action_emb = self.adapter(raw_pert).unsqueeze(1) 

        # 2. Prepare Queries [Batch, Target_Len, Dim]
        queries = self.mask_queries(target_indices) 
        
        # 3. Concatenate Context + Queries
        sequence = torch.cat([context_latents, queries], dim=1)
        
        # 4. Pass through Blocks (Cross-Attn -> Self-Attn -> MLP)
        for block in self.blocks:
            sequence = block(sequence, action_emb)
            
        sequence = self.final_norm(sequence)
        
        # 5. Extract Predictions (corresponding to the Queries)
        predictions = sequence[:, C_Len:, :]  

        # 6. Stochastic Output
        mu = self.head_mu(predictions)
        logvar = self.head_logvar(predictions)
        logvar = torch.clamp(logvar, min=-10, max=2) # Stability clamp
        
        return mu, logvar



@dataclass
class BioJepaConfig:
    num_genes: int = 8192
    n_layer: int = 6
    heads: int = 4
    embed_dim: int = 256
    mlp_ratio: float = 4.0

    # pretraining
    n_pre_layer: int = 3 
    mask_ratio: float = 0.6

    # Loss weights
    sim_coeff: float = 25.0
    std_coeff: float = 25.0
    cov_coeff: float = 1.0
    
class BioJepa(nn.Module):
    def __init__(self, config, pert_embd):
        super().__init__()
        self.config = config

        enc_conf = CellStateEncoderConfig(
            num_genes=config.num_genes,
            n_layer=config.n_layer,
            heads=config.heads,
            embed_dim=config.embed_dim,
            mlp_ratio=config.mlp_ratio
        )
        
        self.student = CellStateEncoder(enc_conf)
        self.teacher = copy.deepcopy(self.student)
        
        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Action Predictor
        pred_conf = ACPredictorConfig(
            num_genes=config.num_genes,
            n_layer=config.n_layer,
            heads=config.heads,
            embed_dim=config.embed_dim,
            mlp_ratio=config.mlp_ratio,
            pert_embd_dim=pert_embd.shape[1]
        )
        self.predictor = ACPredictor(pred_conf, pert_embd)

        ## Pretraining 
        #self.mask_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)

        mask_pred_conf = copy.deepcopy(enc_conf)
        mask_pred_conf.n_layer = config.n_pre_layer
        self.masked_predictor = MaskedPredictor(mask_pred_conf)

    def freeze_encoders(self):
        for p in self.student.parameters():
            p.requires_grad = False

    def vicreg_loss(self, x, y):
        B = x.shape[0]
        num_features = x.shape[-1]
        
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        cov_x = (x.T @ x) / (B - 1)
        cov_y = (y.T @ y) / (B - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features) + \
                   off_diagonal(cov_y).pow_(2).sum().div(num_features)

        return self.config.std_coeff * std_loss + self.config.cov_coeff * cov_loss


    def forward_pretrain(self, x_values, total_counts):
        B, N = x_values.shape


        # Masking
        rand = torch.rand(B, N, device=x_values.device)
        mask_idx = rand < self.config.mask_ratio
        
        # Teacher Target
        with torch.no_grad():
            target_latents = self.teacher(x_values, total_counts)

        # Student
        x_values_student = x_values.clone()
        x_values_student[mask_idx] = 0.0

        context_latents = self.student(x_values_student, total_counts)

        # Predict Missing Latents
        predicted_latents = self.masked_predictor(context_latents)

        
        # Loss Calculation (L1 since mask predictor is deterministic)
        pred_masked = predicted_latents[mask_idx]
        target_masked = target_latents[mask_idx]
        rec_loss = F.l1_loss(pred_masked, target_masked)

        reg_loss = self.vicreg_loss(
            context_latents.reshape(-1, self.config.embed_dim), 
            target_latents.reshape(-1, self.config.embed_dim)
        )

        return self.config.sim_coeff * rec_loss + reg_loss

    def forward(self, x_control, total_control, x_case, total_case, action_id):

        B, N = x_control.shape

        # mask
        rand = torch.rand(B, N, device=x_control.device)
        mask_idx = rand < self.config.mask_ratio
        
        with torch.no_grad():
            # 1. Teacher
            target_latents = self.teacher(x_case, total_case)

            # 2. Student 
            x_input_student = x_control.clone()
            x_input_student[mask_idx] = 0.0
            context_latents = self.student(x_input_student, total_control)
        
        # 3. Predictor 
        target_indices = torch.arange(N, device=x_control.device).expand(B, N)
        pred_mu, pred_logvar = self.predictor(context_latents, action_id, target_indices)
        
        # 4. Loss (Gaussian NLL + VICReg)
        pred_mu_masked = pred_mu[mask_idx]
        pred_logvar_masked = pred_logvar[mask_idx]
        target_masked = target_latents[mask_idx]
        
        rec_loss = F.gaussian_nll_loss(
            pred_mu_masked, 
            target_masked, 
            torch.exp(pred_logvar_masked), 
            reduction='mean'
        )

        reg_loss = self.vicreg_loss(
            pred_mu.reshape(-1, self.config.embed_dim), 
            target_latents.reshape(-1, self.config.embed_dim)
        )

        return self.config.sim_coeff * rec_loss + reg_loss


    @torch.no_grad()
    def update_teacher(self, m=0.996):
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.mul_(m).add_((1 - m) * param_s.data)
            

