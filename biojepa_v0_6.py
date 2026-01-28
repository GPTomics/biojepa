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
        kv_matmul = k.transpose(-2, -1) @ v
        
        # Normalization term (denominator)
        k_sum = k.sum(dim=-2).unsqueeze(-1)
        z = 1.0 / (q @ k_sum + 1e-6)

        # Compute Output (numerator * denominator)
        y = (q @ kv_matmul) * z
        
        # 4. Reassemble
        y = y.transpose(1, 2).contiguous().view(B, T_q, C)
        y = self.c_proj(y)
        
        return y

class GaussianFourierProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.B = nn.Parameter(torch.randn(1, config.embed_dim // 2) * config.gaussian_scale, requires_grad=False)

    def forward(self, x):
        x_proj = (2 * np.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

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
class ActionComposerConfig:
    dna_dim: int = 1536
    protein_dim: int = 320
    chemical_dim: int = 1024
    target_dim: int = 320
    latent_dim: int = 320
    mode_dim: int = 64
    num_modes: int = 9
    max_perts: int = 4

class ActionComposer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        D = config.latent_dim

        # Sequence projectors (modality: 0=dna, 1=protein, 2=chemical)
        self.seq_projectors = nn.ModuleDict({
            'dna': nn.Linear(config.dna_dim, D),
            'protein': nn.Linear(config.protein_dim, D),
            'chemical': nn.Linear(config.chemical_dim, D)
        })
        self.modality_to_key = {0: 'dna', 1: 'protein', 2: 'chemical'}

        # Target projector (always protein ESM-2)
        self.target_projector = nn.Linear(config.target_dim, D)

        # Fusion MLP (concat -> D)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * D, 2 * D),
            nn.GELU(),
            nn.Linear(2 * D, D)
        )

        # Unknown embedding (for when neither seq nor target available)
        self.unknown_embedding = nn.Parameter(torch.randn(1, D) * 0.02)

        # Mode conditioning (FiLM)
        self.mode_embedding = nn.Embedding(config.num_modes, config.mode_dim)
        self.film_scale = nn.Linear(config.mode_dim, D)
        self.film_shift = nn.Linear(config.mode_dim, D)

        # Initialize FiLM to identity
        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)

        # Attention pooling for alignment (query is learned)
        self.pool_query = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.pool_attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True)

    def _encode_sequence(self, seq_emb, modality_id):
        key = self.modality_to_key[modality_id.item() if modality_id.dim() == 0 else modality_id[0].item()]
        proj = self.seq_projectors[key]
        return proj(seq_emb[..., :proj.in_features])

    def _encode_target(self, target_emb):
        return self.target_projector(target_emb)

    def _fuse(self, seq_lat, target_lat, has_seq, has_target):
        B = has_seq.shape[0]
        D = self.config.latent_dim
        device = has_seq.device

        result = torch.zeros(B, D, device=device)

        both_mask = has_seq & has_target
        seq_only_mask = has_seq & ~has_target
        target_only_mask = ~has_seq & has_target
        neither_mask = ~has_seq & ~has_target

        if both_mask.any():
            combined = torch.cat([seq_lat[both_mask], target_lat[both_mask]], dim=-1)
            result[both_mask] = self.fusion_mlp(combined)
        if seq_only_mask.any():
            result[seq_only_mask] = seq_lat[seq_only_mask]
        if target_only_mask.any():
            result[target_only_mask] = target_lat[target_only_mask]
        if neither_mask.any():
            result[neither_mask] = self.unknown_embedding.expand(neither_mask.sum(), -1)

        return result

    def _apply_mode(self, content, mode_ids):
        mode_vecs = self.mode_embedding(mode_ids)
        scale = self.film_scale(mode_vecs)
        shift = self.film_shift(mode_vecs)
        return content * (1.0 + scale) + shift

    def forward(self, seq_emb, target_emb, modality_ids, mode_ids, has_seq, has_target, pert_mask):
        '''
        Args:
            seq_emb: [B, N_pert, max_seq_dim] - sequence embeddings (padded)
            target_emb: [B, N_pert, target_dim] - target embeddings
            modality_ids: [B, N_pert] - 0=dna, 1=protein, 2=chemical
            mode_ids: [B, N_pert] - perturbation mode
            has_seq: [B, N_pert] - bool, whether seq is available
            has_target: [B, N_pert] - bool, whether target is available
            pert_mask: [B, N_pert] - bool, valid perturbations (vs padding)
        Returns:
            action_latents: [B, N_pert, D]
        '''
        B, N_pert = modality_ids.shape
        D = self.config.latent_dim
        device = modality_ids.device

        action_latents = torch.zeros(B, N_pert, D, device=device)

        for p in range(N_pert):
            p_mask = pert_mask[:, p]
            if not p_mask.any():
                continue

            p_has_seq = has_seq[:, p] & p_mask
            p_has_target = has_target[:, p] & p_mask
            p_modality = modality_ids[:, p]
            p_mode = mode_ids[:, p]

            seq_lat = torch.zeros(B, D, device=device)
            target_lat = torch.zeros(B, D, device=device)

            # Encode sequences per modality
            for mod_id in range(3):
                mod_mask = (p_modality == mod_id) & p_has_seq
                if mod_mask.any():
                    proj = self.seq_projectors[self.modality_to_key[mod_id]]
                    seq_lat[mod_mask] = proj(seq_emb[mod_mask, p, :proj.in_features])

            # Encode targets
            if p_has_target.any():
                target_lat[p_has_target] = self.target_projector(target_emb[p_has_target, p])

            # Fuse
            content = self._fuse(seq_lat, target_lat, p_has_seq, p_has_target)

            # Apply mode conditioning
            action = self._apply_mode(content, p_mode)

            action_latents[:, p] = action * p_mask.float().unsqueeze(-1)

        return action_latents

    def encode_sequence_path(self, seq_emb, modality_ids, mode_ids, pert_mask):
        '''Encode using sequence only (for alignment training)'''
        B, N_pert = modality_ids.shape
        D = self.config.latent_dim
        device = modality_ids.device

        action_latents = torch.zeros(B, N_pert, D, device=device)

        for p in range(N_pert):
            p_mask = pert_mask[:, p]
            if not p_mask.any():
                continue

            p_modality = modality_ids[:, p]
            p_mode = mode_ids[:, p]

            seq_lat = torch.zeros(B, D, device=device)
            for mod_id in range(3):
                mod_mask = (p_modality == mod_id) & p_mask
                if mod_mask.any():
                    proj = self.seq_projectors[self.modality_to_key[mod_id]]
                    seq_lat[mod_mask] = proj(seq_emb[mod_mask, p, :proj.in_features])

            action = self._apply_mode(seq_lat, p_mode)
            action_latents[:, p] = action * p_mask.float().unsqueeze(-1)

        return action_latents

    def encode_target_path(self, target_emb, mode_ids, pert_mask):
        '''Encode using target only (for alignment training)'''
        B, N_pert = mode_ids.shape
        D = self.config.latent_dim
        device = mode_ids.device

        action_latents = torch.zeros(B, N_pert, D, device=device)

        for p in range(N_pert):
            p_mask = pert_mask[:, p]
            if not p_mask.any():
                continue

            target_lat = torch.zeros(B, D, device=device)
            target_lat[p_mask] = self.target_projector(target_emb[p_mask, p])

            action = self._apply_mode(target_lat, mode_ids[:, p])
            action_latents[:, p] = action * p_mask.float().unsqueeze(-1)

        return action_latents

    def attention_pool(self, action_latents, pert_mask):
        '''Pool multi-pert action latents to single vector using attention'''
        B, N_pert, D = action_latents.shape

        query = self.pool_query.expand(B, -1, -1)
        key_padding_mask = ~pert_mask

        pooled, _ = self.pool_attn(query, action_latents, action_latents, key_padding_mask=key_padding_mask)
        return pooled.squeeze(1)


@dataclass
class CellStateEncoderConfig:
    num_genes: int = 8192
    n_layer: int = 24 
    heads: int = 12
    embed_dim: int = 768
    mlp_ratio: float = 4.0
    gaussian_scale: float = 2.0
    film_linear_multiple: float = 1.0

class CellStateEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Learnable Gene Embeddings [num_genes, Dim]
        self.gene_embeddings = nn.Parameter(torch.randn(config.num_genes, config.embed_dim) * 0.02)

        # Expression Value Representation
        self.linear_scaler = nn.Linear(1, 1, bias=False)
        self.fourier_input_scaler = nn.Linear(1, 1, bias=False)
        self.fourier_projection = GaussianFourierProjection(config)
        self.film_generator = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim * 2) # Output Gamma + Beta
        )

        # learnable mask token, different than 0 expression
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
        
        # Total Count Injector
        self.total_count_proj = nn.Linear(1, config.embed_dim)

        # Transfomer
        self.blocks = nn.ModuleList([CellStateBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

        # Initiation 
        self.apply(init_weights_robust)
        nn.init.constant_(self.linear_scaler.weight, config.film_linear_multiple)
        nn.init.constant_(self.fourier_input_scaler.weight, 0.1)
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        
    def forward(self, x_values, total_counts, mask_idx=None):
        # 1. Project Genes
        x = x_values.unsqueeze(-1)

        scaled_x = self.linear_scaler(x)
        scaled_x = self.gene_embeddings.unsqueeze(0) * scaled_x

        fourier_x = self.fourier_input_scaler(x)
        fourier_x = self.fourier_projection(fourier_x)
        fourier_x = self.film_generator(fourier_x)
        gamma, beta = torch.chunk(fourier_x, 2, dim=-1)

        x = scaled_x * (1.0 + gamma) + beta

        if mask_idx is not None:
            B, N, D = x.shape
            mask_token_expand = self.mask_token.expand(B, N, D)
            
            x[mask_idx] = mask_token_expand[mask_idx]

        # 3. Total Count Injection
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
    action_dim: int = 320 

class ACPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Perturbation Embedding
        self.adapter = nn.Sequential(
            nn.Linear(config.action_dim, config.embed_dim),
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

    def forward(self, context_latents, action_latents, target_indices):
        '''
        Args:
            context_latents: [B, num_genes, D] - encoded control cell state
            action_latents: [B, N_pert, action_dim] - from ActionComposer (multi-pert)
            target_indices: [B, num_genes] - gene indices for queries
        '''
        B, C_Len, D = context_latents.shape

        # Adapt action latents (handles multi-pert: [B, N_pert, D])
        action_emb = self.adapter(action_latents)

        queries = self.mask_queries(target_indices)
        sequence = torch.cat([context_latents, queries], dim=1)

        for block in self.blocks:
            sequence = block(sequence, action_emb)

        sequence = self.final_norm(sequence)
        predictions = sequence[:, C_Len:, :]

        mu = self.head_mu(predictions)
        logvar = self.head_logvar(predictions)
        logvar = torch.clamp(logvar, min=-10, max=2)

        return mu, logvar



@dataclass
class BioJepaConfig:
    # model size
    num_genes: int = 8192
    n_layer: int = 6
    heads: int = 4
    embed_dim: int = 256
    mlp_ratio: float = 4.0

    # pretraining
    n_pre_layer: int = 3
    mask_ratio: float = 0.6
    gaussian_scale: float = 2.0
    film_linear_multiple: float = 1.0

    # Loss weights
    sim_coeff: float = 25.0
    std_coeff: float = 25.0
    cov_coeff: float = 1.0

    # Perturb Configs
    pert_latent_dim: int = 320
    pert_mode_dim: int = 64

    # EMA
    ema_momentum: float = 0.996
    
class BioJepa(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        enc_conf = CellStateEncoderConfig(
            num_genes=config.num_genes,
            n_layer=config.n_layer,
            heads=config.heads,
            embed_dim=config.embed_dim,
            mlp_ratio=config.mlp_ratio,
            gaussian_scale=config.gaussian_scale,
            film_linear_multiple=config.film_linear_multiple
        )
        
        self.student = CellStateEncoder(enc_conf)
        self.teacher = copy.deepcopy(self.student)
        
        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Action Composer
        composer_conf = ActionComposerConfig(
            latent_dim=config.pert_latent_dim,
            mode_dim=config.pert_mode_dim
        )
        self.composer = ActionComposer(composer_conf)

        # Action Predictor
        pred_conf = ACPredictorConfig(
            num_genes=config.num_genes,
            n_layer=config.n_layer,
            heads=config.heads,
            embed_dim=config.embed_dim,
            mlp_ratio=config.mlp_ratio,
            action_dim=config.pert_latent_dim 
        )
        self.predictor = ACPredictor(pred_conf)

        ## Pretraining 
        mask_pred_conf = copy.deepcopy(enc_conf)
        mask_pred_conf.n_layer = config.n_pre_layer
        self.masked_predictor = MaskedPredictor(mask_pred_conf)

    def freeze_encoders(self):
        for p in self.student.parameters():
            p.requires_grad = False

    def enable_all_gradients(self):
        for p in self.student.parameters():
            p.requires_grad = True
        for p in self.masked_predictor.parameters():
            p.requires_grad = True
        for p in self.composer.parameters():
            p.requires_grad = True
        for p in self.predictor.parameters():
            p.requires_grad = True

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

        context_latents = self.student(x_values_student, total_counts, mask_idx=mask_idx)

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

    def forward_alignment(self, seq_emb, target_emb, modality_ids, mode_ids, pert_mask, temperature=0.07):
        '''
        Dual-path alignment: align sequence representations with target representations.

        Args:
            seq_emb: [B, N_pert, max_seq_dim] - sequence embeddings
            target_emb: [B, N_pert, target_dim] - target protein embeddings
            modality_ids: [B, N_pert] - 0=dna, 1=protein, 2=chemical
            mode_ids: [B, N_pert] - perturbation mode
            pert_mask: [B, N_pert] - valid perturbations
        '''
        z_seq = self.composer.encode_sequence_path(seq_emb, modality_ids, mode_ids, pert_mask)
        z_target = self.composer.encode_target_path(target_emb, mode_ids, pert_mask)

        z_seq = self.composer.attention_pool(z_seq, pert_mask)
        z_target = self.composer.attention_pool(z_target, pert_mask)

        z_seq = F.normalize(z_seq, dim=1)
        z_target = F.normalize(z_target, dim=1)

        logits = torch.matmul(z_seq, z_target.T) / temperature
        labels = torch.arange(logits.shape[0], device=logits.device)

        return F.cross_entropy(logits, labels)

    def forward(self, x_control, total_control, x_case, total_case,
                seq_emb, target_emb, modality_ids, mode_ids, has_seq, has_target, pert_mask):
        '''
        Full training forward with multi-perturbation support.

        Args:
            x_control: [B, num_genes] - control expression
            total_control: [B] - control total counts
            x_case: [B, num_genes] - perturbed expression
            total_case: [B] - perturbed total counts
            seq_emb: [B, N_pert, max_seq_dim] - sequence embeddings
            target_emb: [B, N_pert, target_dim] - target embeddings
            modality_ids: [B, N_pert] - 0=dna, 1=protein, 2=chemical
            mode_ids: [B, N_pert] - perturbation mode
            has_seq: [B, N_pert] - bool, sequence available
            has_target: [B, N_pert] - bool, target available
            pert_mask: [B, N_pert] - bool, valid perturbations
        '''
        B, N = x_control.shape

        rand = torch.rand(B, N, device=x_control.device)
        mask_idx = rand < self.config.mask_ratio

        with torch.no_grad():
            target_latents = self.teacher(x_case, total_case)

            x_input_student = x_control.clone()
            x_input_student[mask_idx] = 0.0
            context_latents = self.student(x_input_student, total_control, mask_idx=mask_idx)

        action_latents = self.composer(seq_emb, target_emb, modality_ids, mode_ids, has_seq, has_target, pert_mask)

        target_indices = torch.arange(N, device=x_control.device).expand(B, N)
        pred_mu, pred_logvar = self.predictor(context_latents, action_latents, target_indices)

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
    def update_teacher(self, m=None):
        if m is None:
            m = self.config.ema_momentum
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.mul_(m).add_((1 - m) * param_s.data)
            

