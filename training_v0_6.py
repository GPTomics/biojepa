import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from biojepa_v0_6 import BioJepa, BioJepaConfig
from evals.linear_expression_decoder import BenchmarkDecoder, BenchmarkDecoderConfig
from config_v0_6 import PretrainConfig, AlignmentConfig, FullTrainingConfig, DecoderConfig, DataConfig, MAX_SEQ_DIM


def create_model(model_cfg: BioJepaConfig, device) -> BioJepa:
    model = BioJepa(model_cfg).to(device)
    return model


def load_feature_banks(data_cfg: DataConfig, device):
    '''Load sequence and target embedding banks for v0.6 multi-pert format.

    Returns:
        seq_banks: dict with 'dna' and optionally 'chemical' embeddings
        target_bank: protein target embeddings
    '''
    seq_banks_dir = Path(data_cfg.data_root) / 'pert_embd' / 'seq_banks'
    target_banks_dir = Path(data_cfg.data_root) / 'pert_embd' / 'target_banks'

    seq_banks = {}

    dna_path = seq_banks_dir / 'dna_embeddings.npy'
    if dna_path.exists():
        seq_banks['dna'] = torch.from_numpy(np.load(dna_path)).float().to(device)
        print(f'Loaded DNA embeddings: {seq_banks["dna"].shape}')

    chem_path = seq_banks_dir / 'chemical_embeddings.npy'
    if chem_path.exists():
        seq_banks['chemical'] = torch.from_numpy(np.load(chem_path)).float().to(device)
        print(f'Loaded chemical embeddings: {seq_banks["chemical"].shape}')

    target_path = target_banks_dir / 'protein_targets.npy'
    target_bank = torch.from_numpy(np.load(target_path)).float().to(device)
    print(f'Loaded target embeddings: {target_bank.shape}')

    return seq_banks, target_bank


def get_seq_embeddings(seq_idx, modality, seq_banks, max_seq_dim=MAX_SEQ_DIM):
    '''Look up sequence embeddings from banks based on modality.

    Args:
        seq_idx: [B, N_pert] indices into modality-specific banks
        modality: [B, N_pert] modality IDs (0=dna, 1=protein, 2=chemical)
        seq_banks: dict with 'dna', 'chemical' tensors
        max_seq_dim: pad all embeddings to this dimension

    Returns:
        seq_emb: [B, N_pert, max_seq_dim] padded embeddings
    '''
    B, N = seq_idx.shape
    device = seq_idx.device
    seq_emb = torch.zeros(B, N, max_seq_dim, device=device)

    for mod_id, mod_key in [(0, 'dna'), (2, 'chemical')]:
        if mod_key not in seq_banks:
            continue
        bank = seq_banks[mod_key]
        mod_mask = (modality == mod_id) & (seq_idx >= 0)
        if mod_mask.any():
            raw_indices = seq_idx[mod_mask]
            in_bounds = raw_indices < bank.shape[0]
            if not in_bounds.all():
                n_oob = (~in_bounds).sum().item()
                print(f'Warning: {n_oob} {mod_key} seq indices out of bounds (max valid={bank.shape[0]-1}), using zeros')
            valid_mask_subset = mod_mask.clone()
            valid_mask_subset[mod_mask] = in_bounds
            if valid_mask_subset.any():
                emb = bank[raw_indices[in_bounds]]
                if emb.shape[-1] < max_seq_dim:
                    emb = F.pad(emb, (0, max_seq_dim - emb.shape[-1]))
                seq_emb[valid_mask_subset] = emb

    return seq_emb


def get_target_embeddings(target_idx, target_bank):
    '''Look up target embeddings from bank.

    Args:
        target_idx: [B, N_pert] indices into target bank
        target_bank: [N_targets, D] target embeddings

    Returns:
        target_emb: [B, N_pert, D] embeddings (zeros for invalid indices)
    '''
    B, N = target_idx.shape
    D = target_bank.shape[-1]
    device = target_idx.device

    target_emb = torch.zeros(B, N, D, device=device)
    valid_mask = target_idx >= 0
    if valid_mask.any():
        raw_indices = target_idx[valid_mask]
        in_bounds = raw_indices < target_bank.shape[0]
        if not in_bounds.all():
            n_oob = (~in_bounds).sum().item()
            print(f'Warning: {n_oob} target indices out of bounds (max valid={target_bank.shape[0]-1}), using zeros')
        final_mask = valid_mask.clone()
        final_mask[valid_mask] = in_bounds
        if final_mask.any():
            target_emb[final_mask] = target_bank[raw_indices[in_bounds]]

    return target_emb


def run_pretraining(model, train_loader, val_loader, cfg: PretrainConfig, device, checkpoint_dir, model_cfg: BioJepaConfig) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = train_loader.total_samples // cfg.batch_size
    if cfg.epochs is not None:
        max_steps = cfg.epochs * steps_per_epoch
    elif cfg.n_steps is not None:
        max_steps = cfg.n_steps
    else:
        raise ValueError('Either epochs or n_steps must be specified')
    print(f'Pretraining: {train_loader.total_samples} samples, {steps_per_epoch} steps/epoch, {max_steps} total steps')
    model.enable_all_gradients()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=max_steps, pct_start=cfg.warmup_pct
    )

    loss_history = []
    total_epoch_loss = 0
    model.train()

    for step in range(max_steps):
        last_step = (step == max_steps - 1)

        if step % 100 == 0 or last_step:
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 25
                for _ in range(val_loss_steps):
                    b = val_loader.next_batch()
                    val_loss = model.forward_pretrain(b.x, b.total)
                    val_loss_accum += val_loss.item()
                avg_val_loss = val_loss_accum / val_loss_steps
                print(f'Step {step} | val loss: {avg_val_loss:.4f}')
            model.train()

        if step > 0 and (step + 1) % steps_per_epoch == 0 and not last_step:
            epoch = (step + 1) // steps_per_epoch
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, checkpoint_dir / f'biojepa_v0_6_pt_epoch_{epoch}.pt')

        b = train_loader.next_batch()
        optimizer.zero_grad()
        loss = model.forward_pretrain(b.x, b.total)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_teacher()
        scheduler.step()

        loss_history.append(loss.item())
        total_epoch_loss += loss.item()

        if step % 25 == 0:
            print(f'Step {step} | Loss: {loss.item():.5f} | LR: {scheduler.get_last_lr()[0]:.2e}')

        if step > 0 and (step + 1) % steps_per_epoch == 0:
            avg_loss = total_epoch_loss / steps_per_epoch
            print(f'=== Epoch {(step + 1) // steps_per_epoch} Done. Avg Loss: {avg_loss:.5f} ===')
            total_epoch_loss = 0

        if last_step:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, checkpoint_dir / f'biojepa_v0_6_pt_final.pt')

    return {'loss_history': loss_history, 'final_loss': loss_history[-1] if loss_history else None}


def run_alignment(model, train_loader, val_loader, seq_banks, target_bank, cfg: AlignmentConfig, device, checkpoint_dir) -> dict:
    '''Run dual-path alignment training.

    The alignment loader provides (seq_idx, target_idx, modality, mode) pairs.
    We train the composer to align sequence embeddings with target embeddings.
    '''
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = train_loader.total_samples // cfg.batch_size
    if cfg.epochs is not None:
        max_steps = cfg.epochs * steps_per_epoch
    elif cfg.n_steps is not None:
        max_steps = cfg.n_steps
    else:
        raise ValueError('Either epochs or n_steps must be specified')
    print(f'Alignment: {train_loader.total_samples} samples, {steps_per_epoch} steps/epoch, {max_steps} total steps')

    for p in model.parameters():
        p.requires_grad = False
    for p in model.composer.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.composer.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=max_steps, pct_start=0.05
    )

    loss_history = []
    model.train()

    for step in range(max_steps):
        last_step = (step == max_steps - 1)

        if step % 500 == 0 or last_step:
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 25
                for _ in range(val_loss_steps):
                    b = val_loader.next_batch()
                    B = b.seq_idx.shape[0]

                    seq_emb = get_seq_embeddings(b.seq_idx.unsqueeze(1), b.modality.unsqueeze(1), seq_banks)
                    target_emb = get_target_embeddings(b.target_idx.unsqueeze(1), target_bank)
                    mode_ids = b.mode.unsqueeze(1)
                    modality_ids = b.modality.unsqueeze(1)
                    pert_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

                    val_loss = model.forward_alignment(seq_emb, target_emb, modality_ids, mode_ids, pert_mask)
                    val_loss_accum += val_loss.item()
                avg_val_loss = val_loss_accum / val_loss_steps
                print(f'Step {step} | val loss: {avg_val_loss:.4f}')
            model.train()

        b = train_loader.next_batch()
        B = b.seq_idx.shape[0]

        seq_emb = get_seq_embeddings(b.seq_idx.unsqueeze(1), b.modality.unsqueeze(1), seq_banks)
        target_emb = get_target_embeddings(b.target_idx.unsqueeze(1), target_bank)
        mode_ids = b.mode.unsqueeze(1)
        modality_ids = b.modality.unsqueeze(1)
        pert_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        optimizer.zero_grad()
        loss = model.forward_alignment(seq_emb, target_emb, modality_ids, mode_ids, pert_mask)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        if step % 100 == 0:
            print(f'Step {step} | Loss: {loss.item():.5f} | LR: {scheduler.get_last_lr()[0]:.2e}')

        if last_step:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, checkpoint_dir / f'biojepa_v0_6_align_final.pt')

    return {'loss_history': loss_history, 'final_loss': loss_history[-1] if loss_history else None}


def run_full_training(model, train_loader, val_loader, seq_banks, target_bank, cfg: FullTrainingConfig, device, checkpoint_dir) -> dict:
    '''Run full action-conditioned training with multi-pert format.'''
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.freeze_encoders()
    for p in model.predictor.parameters():
        p.requires_grad = True
    for p in model.composer.parameters():
        p.requires_grad = True

    steps_per_epoch = train_loader.total_samples // cfg.batch_size
    if cfg.epochs is not None:
        max_steps = cfg.epochs * steps_per_epoch
    elif cfg.n_steps is not None:
        max_steps = cfg.n_steps
    else:
        raise ValueError('Either epochs or n_steps must be specified')
    print(f'Full training: {train_loader.total_samples} samples, {steps_per_epoch} steps/epoch, {max_steps} total steps')

    optimizer = torch.optim.AdamW([
        {'params': model.predictor.parameters(), 'lr': cfg.predictor_lr},
        {'params': model.composer.parameters(), 'lr': cfg.predictor_lr * 0.1}
    ], weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[cfg.predictor_lr, cfg.predictor_lr * 0.1], total_steps=max_steps, pct_start=0.05
    )

    loss_history = []
    total_epoch_loss = 0
    model.train()

    for step in range(max_steps):
        last_step = (step == max_steps - 1)

        if step % 100 == 0 or last_step:
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 25
                for _ in range(val_loss_steps):
                    b = val_loader.next_batch()

                    seq_emb = get_seq_embeddings(b.seq_idx, b.modality, seq_banks)
                    target_emb = get_target_embeddings(b.target_idx, target_bank)
                    pert_mask = torch.arange(b.seq_idx.shape[1], device=device).unsqueeze(0) < b.n_perts.unsqueeze(1)

                    val_loss = model(b.control, b.control_total, b.case, b.case_total,
                                     seq_emb, target_emb, b.modality, b.mode, b.has_seq, b.has_target, pert_mask)
                    val_loss_accum += val_loss.item()
                avg_val_loss = val_loss_accum / val_loss_steps
                print(f'Step {step} | val loss: {avg_val_loss:.4f}')
            model.train()

        if step > 0 and (step + 1) % steps_per_epoch == 0 and not last_step:
            epoch = (step + 1) // steps_per_epoch
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, checkpoint_dir / f'biojepa_v0_6_full_epoch_{epoch}.pt')

        b = train_loader.next_batch()

        seq_emb = get_seq_embeddings(b.seq_idx, b.modality, seq_banks)
        target_emb = get_target_embeddings(b.target_idx, target_bank)
        pert_mask = torch.arange(b.seq_idx.shape[1], device=device).unsqueeze(0) < b.n_perts.unsqueeze(1)

        optimizer.zero_grad()
        loss = model(b.control, b.control_total, b.case, b.case_total,
                     seq_emb, target_emb, b.modality, b.mode, b.has_seq, b.has_target, pert_mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        total_epoch_loss += loss.item()

        if step % 25 == 0:
            print(f'Step {step} | Loss: {loss.item():.5f} | LR: {scheduler.get_last_lr()[0]:.2e}')

        if step > 0 and (step + 1) % steps_per_epoch == 0:
            avg_loss = total_epoch_loss / steps_per_epoch
            print(f'=== Epoch {(step + 1) // steps_per_epoch} Done. Avg Loss: {avg_loss:.5f} ===')
            total_epoch_loss = 0

        if last_step:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, checkpoint_dir / f'biojepa_v0_6_full_final.pt')

    return {'loss_history': loss_history, 'final_loss': loss_history[-1] if loss_history else None}


def train_linear_decoder(model, train_loader, val_loader, seq_banks, target_bank, model_cfg: BioJepaConfig, device, checkpoint_dir, cfg: DecoderConfig) -> tuple[BenchmarkDecoder, dict]:
    '''Train linear decoder on action-conditioned predictions.

    Uses the full prediction pipeline: student encoder -> composer -> predictor.
    Decoder learns to map latent deltas (z_pred - z_context) to expression deltas (xt - xc).
    '''
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    decoder_config = BenchmarkDecoderConfig(embed_dim=model_cfg.embed_dim)
    decoder = BenchmarkDecoder(decoder_config).to(device)

    steps_per_epoch = train_loader.total_samples // train_loader.batch_size
    if cfg.epochs is not None:
        max_steps = cfg.epochs * steps_per_epoch
    elif cfg.n_steps is not None:
        max_steps = cfg.n_steps
    else:
        raise ValueError('Either epochs or n_steps must be specified')

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=max_steps, pct_start=0.05)

    loss_fn = nn.MSELoss()
    loss_history = []

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    decoder.train()

    for step in range(max_steps):
        last_step = (step == max_steps - 1)

        if step % 100 == 0 or last_step:
            decoder.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 25
                for _ in range(val_loss_steps):
                    b = val_loader.next_batch()
                    B, N = b.control.shape

                    seq_emb = get_seq_embeddings(b.seq_idx, b.modality, seq_banks)
                    target_emb = get_target_embeddings(b.target_idx, target_bank)
                    pert_mask = torch.arange(b.seq_idx.shape[1], device=device).unsqueeze(0) < b.n_perts.unsqueeze(1)

                    z_context = model.student(b.control, b.control_total, mask_idx=None)
                    action_latents = model.composer(seq_emb, target_emb, b.modality, b.mode, b.has_seq, b.has_target, pert_mask)
                    target_indices = torch.arange(N, device=device).expand(B, N)
                    z_pred_mu, _ = model.predictor(z_context, action_latents, target_indices)

                    pred_delta = decoder(z_pred_mu) - decoder(z_context)
                    real_delta = b.case - b.control
                    val_loss = loss_fn(pred_delta, real_delta)
                    val_loss_accum += val_loss.item()
                avg_val_loss = val_loss_accum / val_loss_steps
                print(f'Decoder Step {step} | val loss: {avg_val_loss:.4f}')
            decoder.train()

        b = train_loader.next_batch()
        B, N = b.control.shape

        seq_emb = get_seq_embeddings(b.seq_idx, b.modality, seq_banks)
        target_emb = get_target_embeddings(b.target_idx, target_bank)
        pert_mask = torch.arange(b.seq_idx.shape[1], device=device).unsqueeze(0) < b.n_perts.unsqueeze(1)

        with torch.no_grad():
            z_context = model.student(b.control, b.control_total, mask_idx=None)
            action_latents = model.composer(seq_emb, target_emb, b.modality, b.mode, b.has_seq, b.has_target, pert_mask)
            target_indices = torch.arange(N, device=device).expand(B, N)
            z_pred_mu, _ = model.predictor(z_context, action_latents, target_indices)

        optimizer.zero_grad()
        pred_delta = decoder(z_pred_mu) - decoder(z_context)
        real_delta = b.case - b.control
        loss = loss_fn(pred_delta, real_delta)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        if step % 25 == 0:
            print(f'Decoder Step {step} | Loss: {loss.item():.5f}')

        if last_step:
            torch.save({
                'model': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, checkpoint_dir / f'biojepa_v0_6_decoder_final.pt')

    return decoder, {'loss_history': loss_history, 'final_loss': loss_history[-1] if loss_history else None}
