import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from dataclasses import dataclass

torch.manual_seed(1337)


# utils
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.float()
        norm = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).type_as(x)

def _make_divisible(v, divisor=64):
    return max(divisor, int(v + divisor / 2) // divisor * divisor)

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
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)
    elif isinstance(module, BioLinearAttention):
        nn.init.zeros_(module.gate.weight)
        nn.init.constant_(module.gate.bias, 2.0)

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
        self.gate = nn.Linear(config.embed_dim, config.embed_dim)

    def forward(self, x, kv=None):
        B, T_q, C = x.size()
        kv_input = kv if kv is not None else x
        T_kv = kv_input.size(1)

        q = self.q_proj(x).view(B, T_q, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, T_kv, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, T_kv, self.heads, self.head_dim).transpose(1, 2)

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        kv_matmul = k.transpose(-2, -1) @ v
        k_sum = k.sum(dim=-2).unsqueeze(-1)
        z = 1.0 / (q @ k_sum + 1e-6)
        y = (q @ kv_matmul) * z

        y = y.transpose(1, 2).contiguous().view(B, T_q, C)
        if kv is None:
            y = torch.sigmoid(self.gate(x)) * y
        y = self.c_proj(y)

        return y

class GaussianFourierProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.B = nn.Parameter(torch.randn(1, config.embed_dim // 2) * config.gaussian_scale, requires_grad=False)

    def forward(self, x):
        x_proj = (2 * np.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = _make_divisible(int(config.embed_dim * config.mlp_ratio * 2 / 3))
        self.w1 = nn.Linear(config.embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.embed_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config.embed_dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class CellStateBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.embed_dim)
        self.attn = BioLinearAttention(config)
        self.ln_2 = RMSNorm(config.embed_dim)
        self.mlp = SwiGLU(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class PredictorBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.embed_dim)
        self.action_attn = BioLinearAttention(config)

        self.ln_2 = RMSNorm(config.embed_dim)
        self.self_attn = BioLinearAttention(config)

        self.ln_3 = RMSNorm(config.embed_dim)
        self.mlp = SwiGLU(config)

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

        self.blocks = nn.ModuleList([CellStateBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.embed_dim)
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
    heads: int = None

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

        # Unknown embedding (for when neither seq nor target available)
        self.unknown_embedding = nn.Parameter(torch.randn(1, D) * 0.02)

        # Mode conditioning (FiLM)
        self.mode_embedding = nn.Embedding(config.num_modes, config.mode_dim)
        self.film_scale = nn.Linear(config.mode_dim, D)
        self.film_shift = nn.Linear(config.mode_dim, D)

        # latent masking
        self.latent_dropout = nn.Dropout(p=0.15)

        # Initialize FiLM to identity
        nn.init.normal_(self.film_scale.weight, std=0.02) #add some noise
        nn.init.zeros_(self.film_scale.bias)
        nn.init.normal_(self.film_shift.weight, std=0.02) #add some noise
        nn.init.zeros_(self.film_shift.bias)

        # Attention pooling for alignment (query is learned)
        self.pool_query = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.pool_attn = nn.MultiheadAttention(D, num_heads=config.heads, batch_first=True)

        self.dose_proj = nn.Sequential(nn.Linear(1, D), nn.SiLU(), nn.Linear(D, D))
        nn.init.zeros_(self.dose_proj[2].weight)
        nn.init.ones_(self.dose_proj[2].bias)

    def _encode_target(self, target_emb):
        return self.target_projector(target_emb)

    def _fuse(self, seq_lat, target_lat, has_seq, has_target):
        seq_masked = self.latent_dropout(seq_lat)
        target_masked = self.latent_dropout(target_lat)
        
        result = seq_masked + target_masked

        neither_mask = ~(has_seq | has_target)

        if neither_mask.any():
            result[neither_mask] = self.unknown_embedding.expand(neither_mask.sum(), -1)

        return result

    def _apply_mode(self, content, mode_ids):
        mode_ids = mode_ids.clamp(0, self.config.num_modes - 1)
        mode_vecs = self.mode_embedding(mode_ids)
        scale = self.film_scale(mode_vecs)
        shift = self.film_shift(mode_vecs)
        return content * (1.0 + scale) + shift

    def _apply_dose(self, action, p_dose, p_mask):
        dose_valid = (p_dose != -1.0) & p_mask
        dose_scale = self.dose_proj(p_dose.unsqueeze(-1))
        dose_scale = torch.where(dose_valid.unsqueeze(-1), dose_scale, torch.ones_like(dose_scale))
        return action * dose_scale

    @torch.autocast('cuda', enabled=False)
    def forward(self, seq_emb, target_emb, modality_ids, mode_ids, has_seq, has_target, pert_mask, dose=None):
        '''
        Args:
            seq_emb: - sequence embeddings (padded)
            target_emb: - target embeddings
            modality_ids: - 0=dna, 1=protein, 2=chemical
            mode_ids: - perturbation mode
            has_seq: - bool, whether seq is available
            has_target: - bool, whether target is available
            pert_mask: - bool, valid perturbations (vs padding)
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

            if dose is not None:
                action = self._apply_dose(action, dose[:, p], p_mask)

            action_latents[:, p] = action * p_mask.float().unsqueeze(-1)

        return action_latents

    @torch.autocast('cuda', enabled=False)
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

    @torch.autocast('cuda', enabled=False)
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
        self.expr_scaler = nn.Linear(1, 1, bias=False)
        self.fourier_input_scaler = nn.Linear(1, 1, bias=False)
        self.fourier_projection = GaussianFourierProjection(config)
        self.film_generator = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim * 2) # Output Gamma + Beta
        )

        self.mask_token = nn.Parameter(torch.randn(config.embed_dim) * 0.02)
        self.unknown_token = nn.Parameter(torch.randn(config.embed_dim) * 0.02)

        # Total Count Injector
        self.total_count_proj = nn.Linear(1, config.embed_dim)

        self.blocks = nn.ModuleList([CellStateBlock(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.embed_dim)

        self.apply(init_weights_robust)
        nn.init.constant_(self.expr_scaler.weight, config.film_linear_multiple)
        nn.init.constant_(self.fourier_input_scaler.weight, 0.1)
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        
    def forward(self, x_values, total_counts, mask_idx=None, unknown_mask=None):
        x = x_values.unsqueeze(-1)

        scaled_x = self.expr_scaler(x)
        scaled_x = self.gene_embeddings.unsqueeze(0) * scaled_x

        fourier_x = self.fourier_input_scaler(x)
        fourier_x = self.fourier_projection(fourier_x)
        fourier_x = self.film_generator(fourier_x)
        gamma, beta = torch.chunk(fourier_x, 2, dim=-1)

        x = scaled_x * (1.0 + gamma) + beta
        x = self.gene_embeddings.unsqueeze(0) + x

        if mask_idx is not None:
            x = torch.where(mask_idx.unsqueeze(-1), self.mask_token, x)
        if unknown_mask is not None:
            x = torch.where(unknown_mask.unsqueeze(-1), self.unknown_token, x)

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

        self.adapter = nn.Sequential(
            nn.Linear(config.action_dim, config.embed_dim),
            RMSNorm(config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim)
        )

        self.mask_queries = nn.Embedding(config.num_genes, config.embed_dim)
        self.blocks = nn.ModuleList([PredictorBlock(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.embed_dim)

        self.head_mu = nn.Linear(config.embed_dim, config.embed_dim)
        self.head_logvar = nn.Linear(config.embed_dim, config.embed_dim)

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
        with torch.autocast(device_type='cuda', enabled=False):
            logvar = self.head_logvar(predictions.float())
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
    ema_momentum: float = 0.995

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
            mode_dim=config.pert_mode_dim,
            heads=config.heads
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

        self.null_action = nn.Parameter(torch.zeros(1, 1, config.pert_latent_dim))

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
        self.student.fourier_projection.B.requires_grad = False

    def enable_encoder_gradients(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.student.parameters():
            p.requires_grad = True
        for p in self.masked_predictor.parameters():
            p.requires_grad = True
        self.student.fourier_projection.B.requires_grad = False

    def vicreg_loss(self, x, y, return_components=False):
        x, y = x.float(), y.float()
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

        total = self.config.std_coeff * std_loss + self.config.cov_coeff * cov_loss
        if return_components:
            return total, {'std_loss': std_loss, 'cov_loss': cov_loss}
        return total


    def forward_encoder(self, x_values, total_counts, gene_mask=None, context_coeff=0.0, return_components=False):
        B, N = x_values.shape

        rand = torch.rand(B, N, device=x_values.device)
        mask_idx = rand < self.config.mask_ratio

        unknown_mask = ~gene_mask if gene_mask is not None else None

        with torch.no_grad():
            target_latents = self.teacher(x_values, total_counts, unknown_mask=unknown_mask)

        x_values_student = x_values.clone()
        x_values_student[mask_idx] = 0.0

        context_latents = self.student(x_values_student, total_counts, mask_idx=mask_idx, unknown_mask=unknown_mask)
        predicted_latents = self.masked_predictor(context_latents)

        if gene_mask is not None:
            is_masked = mask_idx & gene_mask
            is_context = ~mask_idx & gene_mask
        else:
            is_masked = mask_idx
            is_context = ~mask_idx

        rec_loss = F.l1_loss(predicted_latents[is_masked], target_latents[is_masked])

        if context_coeff > 0 and is_context.any():
            context_loss = F.mse_loss(predicted_latents[is_context], target_latents[is_context])
        else:
            context_loss = torch.tensor(0.0, device=x_values.device)

        reg_loss, reg_components = self.vicreg_loss(
            context_latents.reshape(-1, self.config.embed_dim),
            target_latents.reshape(-1, self.config.embed_dim),
            return_components=True
        )

        total = self.config.sim_coeff * rec_loss + context_coeff * context_loss + reg_loss

        if return_components:
            return total, {'l1_masked': rec_loss, 'context_l2': context_loss, **reg_components}
        return total

    def forward_composer(self, seq_emb, target_emb, modality_ids, mode_ids, pert_mask, temperature=0.012):
        '''Dual-path alignment: align sequence representations with target representations (seq-to-target InfoNCE).'''
        z_seq = self.composer.encode_sequence_path(seq_emb, modality_ids, mode_ids, pert_mask)
        z_target = self.composer.encode_target_path(target_emb, mode_ids, pert_mask)

        z_seq = self.composer.attention_pool(z_seq, pert_mask)
        z_target = self.composer.attention_pool(z_target, pert_mask)

        z_seq = F.normalize(z_seq, dim=1)
        z_target = F.normalize(z_target, dim=1)

        logits = torch.matmul(z_seq, z_target.T) / temperature
        labels = torch.arange(z_seq.shape[0], device=z_seq.device)
        return F.cross_entropy(logits, labels)

    def forward(self, x_control, total_control, x_case, total_case,
                seq_emb, target_emb, modality_ids, mode_ids, has_seq, has_target, pert_mask,
                mask_ratio=None, beta_nll=0.0, return_components=False, p_uncond=0.0, unknown_mask=None, dose=None):
        '''AC training forward with multi-perturbation support.'''
        B, N = x_control.shape

        effective_mask_ratio = mask_ratio if mask_ratio is not None else self.config.mask_ratio
        rand = torch.rand(B, N, device=x_control.device)
        mask_idx = rand < effective_mask_ratio

        with torch.no_grad():
            target_latents = self.teacher(x_case, total_case, unknown_mask=unknown_mask)

            x_input_student = x_control.clone()
            x_input_student[mask_idx] = 0.0
            context_latents = self.student(x_input_student, total_control, mask_idx=mask_idx, unknown_mask=unknown_mask)

        action_latents = self.composer(seq_emb, target_emb, modality_ids, mode_ids, has_seq, has_target, pert_mask, dose=dose)

        if self.training and p_uncond > 0 and torch.rand(1).item() < p_uncond:
            action_latents = self.null_action.expand(B, action_latents.shape[1], -1)

        target_indices = torch.arange(N, device=x_control.device).expand(B, N)
        pred_mu, pred_logvar = self.predictor(context_latents, action_latents, target_indices)

        with torch.autocast(device_type='cuda', enabled=False):
            variance = torch.exp(pred_logvar)
            nll = F.gaussian_nll_loss(pred_mu.float(), target_latents.float(), variance, reduction='none')
            if unknown_mask is not None:
                measured = ~unknown_mask
                nll = nll[measured]
                variance = variance[measured]
            rec_loss = (nll * variance.detach().pow(beta_nll)).mean() if beta_nll > 0 else nll.mean()

        if return_components:
            reg_loss, reg_components = self.vicreg_loss(
                pred_mu.reshape(-1, self.config.embed_dim),
                target_latents.reshape(-1, self.config.embed_dim),
                return_components=True
            )
            total = self.config.sim_coeff * rec_loss + reg_loss
            return total, {'nll': rec_loss, **reg_components}

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
            

