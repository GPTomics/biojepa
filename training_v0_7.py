import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import gc
import random
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import LambdaLR

from biojepa_v0_7 import BioJepa
from evals.evals import EvalContext, run_encoder_evals, summarize_encoder_evals
from evals.linear_expression_decoder import BenchmarkDecoder, BenchmarkDecoderConfig
from config_v0_7 import MAX_SEQ_DIM, VERSION


def reset_seed(seed=1337):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_model(model_cfg, device):
    model = BioJepa(model_cfg).to(device)
    return model


def maybe_compile(model, compile_model=False, mode='max-autotune-no-cudagraphs'):
    if not (compile_model and torch.cuda.is_available() and hasattr(torch, 'compile')):
        return model
    try:
        model.student = torch.compile(model.student, mode=mode)
        model.teacher = torch.compile(model.teacher, mode=mode)
        model.predictor = torch.compile(model.predictor, mode=mode)
        model.masked_predictor = torch.compile(model.masked_predictor, mode=mode)
    except Exception as e:
        print(f'torch.compile failed: {e}, using eager mode')
    return model


def load_feature_banks(data_cfg, device):
    seq_banks_dir = Path(data_cfg.data_root) / 'pert_embd' / 'seq_banks'
    target_banks_dir = Path(data_cfg.data_root) / 'pert_embd' / 'target_banks'

    seq_banks = {}

    dna_path = seq_banks_dir / 'dna_embeddings.npy'
    if dna_path.exists():
        seq_banks['dna'] = torch.from_numpy(np.load(dna_path)).float().to(device)
        print(f'Loaded DNA embeddings: {seq_banks["dna"].shape}')

    chem_path = seq_banks_dir / 'chemical_embeddings.npy'
    if chem_path.exists():
        chem = torch.from_numpy(np.load(chem_path)).float().to(device)
        if chem.shape[-1] < MAX_SEQ_DIM:
            print(f'Warning: Chemical embeddings need padding ({chem.shape[-1]} -> {MAX_SEQ_DIM}). '
                  f'Run data_prep/prepad_embeddings.py for permanent fix.')
            chem = F.pad(chem, (0, MAX_SEQ_DIM - chem.shape[-1]))
        seq_banks['chemical'] = chem
        print(f'Loaded chemical embeddings: {seq_banks["chemical"].shape}')

    target_path = target_banks_dir / 'protein_targets.npy'
    target_bank = torch.from_numpy(np.load(target_path)).float().to(device)
    print(f'Loaded target embeddings: {target_bank.shape}')

    return seq_banks, target_bank


def get_seq_embeddings(seq_idx, modality, seq_banks, max_seq_dim=MAX_SEQ_DIM):
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


def _make_writer(log_dir, stage_name):
    if log_dir is None:
        return None
    return SummaryWriter(Path(log_dir) / stage_name)


def _wsd_lambda(step, warmup_steps, phase2_start_step, max_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    if step < phase2_start_step:
        return 1.0
    decay_steps = max_steps - phase2_start_step
    return max(0.0, 1.0 - (step - phase2_start_step) / max(1, decay_steps))


def run_encoder_training(model, train_loader, val_loader, cfg, device, data_cfg, model_cfg, use_amp=False, use_fused_optimizer=False, eval_every_n_epochs=None, log_dir='default'):
    reset_seed()
    checkpoint_dir = Path(data_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if log_dir == 'default':
        log_dir = checkpoint_dir / 'training_logs'
    writer = _make_writer(log_dir, 'encoder')

    use_autocast = use_amp and device.type == 'cuda'
    fused = use_fused_optimizer and torch.cuda.is_available()

    steps_per_epoch = train_loader.total_samples // cfg.batch_size
    if cfg.epochs is not None:
        max_steps = cfg.epochs * steps_per_epoch
    elif cfg.n_steps is not None:
        max_steps = cfg.n_steps
    else:
        raise ValueError('Either epochs or n_steps must be specified')
    print(f'Encoder training: {train_loader.total_samples} samples, {steps_per_epoch} steps/epoch, {max_steps} total steps')
    model.enable_encoder_gradients()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay, fused=fused)

    phase2_start_step = int(max_steps * cfg.phase2_start_pct)
    warmup_steps = int(max_steps * cfg.warmup_pct)
    scheduler = LambdaLR(optimizer, lambda step: _wsd_lambda(step, warmup_steps, phase2_start_step, max_steps))
    print(f'WSD schedule: warmup={warmup_steps} steps, phase2 starts at step {phase2_start_step}')

    context_ramp_start = int(max_steps * (1.0 - cfg.context_ramp_pct))
    if cfg.context_coeff > 0:
        print(f'Context L2: target={cfg.context_coeff}, ramp starts at step {context_ramp_start}')
    if cfg.ema_final_momentum is not None:
        print(f'EMA annealing: {model_cfg.ema_momentum} -> {cfg.ema_final_momentum} during phase 2')

    loss_history = []
    epoch_evals = {}
    total_epoch_loss = 0
    model.train()

    for step in range(max_steps):
        last_step = (step == max_steps - 1)

        if cfg.context_coeff > 0 and step >= context_ramp_start:
            progress = (step - context_ramp_start) / max(1, max_steps - 1 - context_ramp_start)
            current_context_coeff = cfg.context_coeff * progress
        else:
            current_context_coeff = 0.0

        if step % 500 == 0 or last_step:
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 25
                for _ in range(val_loss_steps):
                    b = val_loader.next_batch()
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
                        val_loss = model.forward_encoder(b.x, b.total, gene_mask=b.gene_mask, context_coeff=current_context_coeff)
                    val_loss_accum += val_loss.item()
                avg_val_loss = val_loss_accum / val_loss_steps
                print(f'Step {step} | val loss: {avg_val_loss:.4f}')
                if writer:
                    writer.add_scalar('loss/val', avg_val_loss, step)
            model.train()

        if step > 0 and (step % 10000 == 0 or (step + 1) % steps_per_epoch == 0) and not last_step:
            epoch = (step + 1) // steps_per_epoch
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, checkpoint_dir / f'biojepa_{VERSION}_encoder_epoch_{epoch}_step{step}.pt')

        b = train_loader.next_batch()
        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
            if writer:
                loss, components = model.forward_encoder(b.x, b.total, gene_mask=b.gene_mask, context_coeff=current_context_coeff, return_components=True)
            else:
                loss = model.forward_encoder(b.x, b.total, gene_mask=b.gene_mask, context_coeff=current_context_coeff)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(trainable_params, 5.0)
        optimizer.step()

        if cfg.ema_final_momentum is not None and step >= phase2_start_step:
            ema_progress = (step - phase2_start_step) / max(1, max_steps - 1 - phase2_start_step)
            current_ema = model_cfg.ema_momentum + (cfg.ema_final_momentum - model_cfg.ema_momentum) * ema_progress
            model.update_teacher(m=current_ema)
        else:
            model.update_teacher()
        scheduler.step()

        loss_history.append(loss.item())
        total_epoch_loss += loss.item()

        current_phase = 2 if step >= phase2_start_step else 1
        if writer:
            writer.add_scalar('loss/train', loss.item(), step)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], step)
            writer.add_scalar('grad_norm', grad_norm.item(), step)
            writer.add_scalar('phase', current_phase, step)
            writer.add_scalar('context_coeff', current_context_coeff, step)
            if cfg.ema_final_momentum is not None and step >= phase2_start_step:
                writer.add_scalar('ema_momentum', current_ema, step)
            for k, v in components.items():
                writer.add_scalar(f'loss/{k}', v.item(), step)

        if step % 100 == 0:
            print(f'Step {step} | Loss: {loss.item():.5f} | LR: {scheduler.get_last_lr()[0]:.2e} | Phase: {current_phase}')

        if step > 0 and (step + 1) % steps_per_epoch == 0:
            avg_loss = total_epoch_loss / steps_per_epoch
            print(f'=== Epoch {(step + 1) // steps_per_epoch} Done. Avg Loss: {avg_loss:.5f} ===')
            total_epoch_loss = 0

            epoch = (step + 1) // steps_per_epoch
            if eval_every_n_epochs and data_cfg.eval_results_dir and epoch % eval_every_n_epochs == 0:
                print(f'--- Running epoch {epoch} evals ---')
                eval_config = {
                    'num_genes': model_cfg.num_genes, 'embed_dim': model_cfg.embed_dim,
                    'n_layer': model_cfg.n_layer, 'heads': model_cfg.heads,
                    'batch_size': cfg.batch_size, 'verbose': False,
                    'eval_split': 'val',
                }
                ckpt_name = f'biojepa_{VERSION}_encoder_final.pt' if last_step else f'biojepa_{VERSION}_encoder_epoch_{epoch}_step{step}.pt'
                model.eval()
                eval_ctx = EvalContext(config=eval_config, data_root=data_cfg.data_root, checkpoint_root=data_cfg.data_root, ref_dir=data_cfg.ref_dir)
                eval_ctx._biojepa = model
                try:
                    raw_results = run_encoder_evals(eval_ctx)
                    metrics = summarize_encoder_evals(raw_results)
                    epoch_evals[epoch] = {'step': step + 1, 'avg_loss': round(avg_loss, 5), 'checkpoint': ckpt_name, 'metrics': metrics}
                    print(f'Epoch {epoch} metrics: {metrics}')
                    eval_results_path = Path(data_cfg.eval_results_dir)
                    eval_results_path.mkdir(parents=True, exist_ok=True)
                    (eval_results_path / 'encoder_epoch_evals.json').write_text(json.dumps(epoch_evals, indent=2))
                    if writer:
                        for k, v in metrics.items():
                            writer.add_scalar(f'eval/{k}', v, epoch)
                finally:
                    eval_ctx._biojepa = None
                    del eval_ctx
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    model.train()

        if last_step:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, checkpoint_dir / f'biojepa_{VERSION}_encoder_final.pt')

    if writer:
        writer.close()

    return {'loss_history': loss_history, 'epoch_evals': epoch_evals, 'final_loss': loss_history[-1] if loss_history else None}


def run_composer_training(model, train_loader, val_loader, seq_banks, target_bank, cfg, device, checkpoint_dir, use_amp=False, use_fused_optimizer=False, log_dir='default'):
    reset_seed()
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if log_dir == 'default':
        log_dir = checkpoint_dir / 'training_logs'
    writer = _make_writer(log_dir, 'composer')

    use_autocast = use_amp and device.type == 'cuda'
    fused = use_fused_optimizer and torch.cuda.is_available()

    steps_per_epoch = train_loader.total_samples // cfg.batch_size
    if cfg.epochs is not None:
        max_steps = cfg.epochs * steps_per_epoch
    elif cfg.n_steps is not None:
        max_steps = cfg.n_steps
    else:
        raise ValueError('Either epochs or n_steps must be specified')
    print(f'Composer training: {train_loader.total_samples} samples, {steps_per_epoch} steps/epoch, {max_steps} total steps')
    print(f'InfoNCE loss (temperature: {cfg.temperature})')

    for p in model.parameters():
        p.requires_grad = False
    for p in model.composer.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.composer.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=fused)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=max_steps, pct_start=0.05)

    loss_history = []
    model.train()

    for step in range(max_steps):
        last_step = (step == max_steps - 1)

        if step % 10000 == 0 or last_step:
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 50
                for _ in range(val_loss_steps):
                    b = val_loader.next_batch()
                    B = b.seq_idx.shape[0]

                    seq_emb = get_seq_embeddings(b.seq_idx.unsqueeze(1), b.modality.unsqueeze(1), seq_banks)
                    target_emb = get_target_embeddings(b.target_idx.unsqueeze(1), target_bank)
                    mode_ids = b.mode.unsqueeze(1)
                    modality_ids = b.modality.unsqueeze(1)
                    pert_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
                        val_loss = model.forward_composer(seq_emb, target_emb, modality_ids, mode_ids, pert_mask, temperature=cfg.temperature)
                    val_loss_accum += val_loss.item()
                avg_val_loss = val_loss_accum / val_loss_steps
                print(f'Step {step} | val loss: {avg_val_loss:.4f}')
                if writer:
                    writer.add_scalar('loss/val', avg_val_loss, step)
            model.train()

        b = train_loader.next_batch()
        B = b.seq_idx.shape[0]

        seq_emb = get_seq_embeddings(b.seq_idx.unsqueeze(1), b.modality.unsqueeze(1), seq_banks)
        target_emb = get_target_embeddings(b.target_idx.unsqueeze(1), target_bank)
        mode_ids = b.mode.unsqueeze(1)
        modality_ids = b.modality.unsqueeze(1)
        pert_mask = torch.ones(B, 1, dtype=torch.bool, device=device)

        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
            loss = model.forward_composer(seq_emb, target_emb, modality_ids, mode_ids, pert_mask, temperature=cfg.temperature)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        if writer:
            writer.add_scalar('loss/train', loss.item(), step)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], step)

        if step % 2500 == 0:
            print(f'Step {step} | Loss: {loss.item():.5f} | LR: {scheduler.get_last_lr()[0]:.2e}')

        if last_step:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, checkpoint_dir / f'biojepa_{VERSION}_composer_final.pt')

    if writer:
        writer.close()

    return {'loss_history': loss_history, 'final_loss': loss_history[-1] if loss_history else None}


def get_mask_ratio(step, base_mask_ratio, anneal_start_step, max_steps, floor=0.1):
    if step < anneal_start_step:
        return base_mask_ratio
    progress = (step - anneal_start_step) / max(1, max_steps - 1 - anneal_start_step)
    return base_mask_ratio + (floor - base_mask_ratio) * progress


def get_beta_nll(step, target_beta, anneal_steps):
    if anneal_steps <= 0 or target_beta == 0:
        return target_beta
    return target_beta * min(1.0, step / max(1, anneal_steps))


def run_ac_training(model, train_loader, val_loader, seq_banks, target_bank, cfg, device, checkpoint_dir, use_amp=False, use_fused_optimizer=False, log_dir='default'):
    reset_seed()
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if log_dir == 'default':
        log_dir = checkpoint_dir / 'training_logs'
    writer = _make_writer(log_dir, 'ac')

    use_autocast = use_amp and device.type == 'cuda'
    fused = use_fused_optimizer and torch.cuda.is_available()

    model.freeze_encoders()
    for p in model.masked_predictor.parameters():
        p.requires_grad = False
    for p in model.predictor.parameters():
        p.requires_grad = True
    for p in model.composer.parameters():
        p.requires_grad = True
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    steps_per_epoch = train_loader.total_samples // cfg.batch_size
    if cfg.epochs is not None:
        max_steps = cfg.epochs * steps_per_epoch
    elif cfg.n_steps is not None:
        max_steps = cfg.n_steps
    else:
        raise ValueError('Either epochs or n_steps must be specified')
    print(f'AC training: {train_loader.total_samples} samples, {steps_per_epoch} steps/epoch, {max_steps} total steps')

    optimizer = torch.optim.AdamW([
        {'params': model.predictor.parameters(), 'lr': cfg.predictor_lr},
        {'params': list(model.composer.parameters()), 'lr': cfg.predictor_lr * cfg.composer_lr_mult}
    ], weight_decay=cfg.weight_decay, fused=fused)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[cfg.predictor_lr, cfg.predictor_lr * cfg.composer_lr_mult], total_steps=max_steps, pct_start=0.05
    )

    base_mask_ratio = model.config.mask_ratio
    if cfg.mask_anneal_pct > 0:
        anneal_start_step = max(0, int(max_steps * (1 - cfg.mask_anneal_pct)))
    else:
        anneal_start_step = max_steps
    print(f'Mask annealing: starts at step {anneal_start_step} ({base_mask_ratio:.3f} -> {cfg.mask_anneal_floor:.3f})')

    target_beta = cfg.beta_nll_target
    beta_anneal_pct = cfg.beta_nll_anneal_pct
    beta_anneal_steps = int(max_steps * beta_anneal_pct)
    print(f'Beta-NLL: target={target_beta:.2f}, anneal steps={beta_anneal_steps}')

    loss_history = []
    total_epoch_loss = 0
    model.train()

    for step in range(max_steps):
        last_step = (step == max_steps - 1)
        current_mask_ratio = get_mask_ratio(step, base_mask_ratio, anneal_start_step, max_steps, floor=cfg.mask_anneal_floor)
        current_beta = get_beta_nll(step, target_beta, beta_anneal_steps)

        if step % 500 == 0 or last_step:
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 25
                for _ in range(val_loss_steps):
                    b = val_loader.next_batch()

                    seq_emb = get_seq_embeddings(b.seq_idx, b.modality, seq_banks)
                    target_emb = get_target_embeddings(b.target_idx, target_bank)
                    pert_mask = torch.arange(b.seq_idx.shape[1], device=device).unsqueeze(0) < b.n_perts.unsqueeze(1)

                    unknown_mask = ~b.gene_mask

                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
                        val_loss = model(b.control, b.control_total, b.case, b.case_total,
                                         seq_emb, target_emb, b.modality, b.mode, b.has_seq, b.has_target, pert_mask,
                                         mask_ratio=current_mask_ratio, beta_nll=current_beta, unknown_mask=unknown_mask, dose=b.dose)
                    val_loss_accum += val_loss.item()
                avg_val_loss = val_loss_accum / val_loss_steps
                print(f'Step {step} | val loss: {avg_val_loss:.4f}')
                if writer:
                    writer.add_scalar('loss/val', avg_val_loss, step)
            model.train()

        if step > 0 and (step % 10000 == 0 or (step + 1) % steps_per_epoch == 0) and not last_step:
            epoch = (step + 1) // steps_per_epoch
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, checkpoint_dir / f'biojepa_{VERSION}_ac_epoch_{epoch}_step{step}.pt')

        b = train_loader.next_batch()

        seq_emb = get_seq_embeddings(b.seq_idx, b.modality, seq_banks)
        target_emb = get_target_embeddings(b.target_idx, target_bank)
        pert_mask = torch.arange(b.seq_idx.shape[1], device=device).unsqueeze(0) < b.n_perts.unsqueeze(1)
        unknown_mask = ~b.gene_mask

        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
            if writer:
                loss, components = model(b.control, b.control_total, b.case, b.case_total,
                                         seq_emb, target_emb, b.modality, b.mode, b.has_seq, b.has_target, pert_mask,
                                         mask_ratio=current_mask_ratio, beta_nll=current_beta, return_components=True, unknown_mask=unknown_mask, dose=b.dose)
            else:
                loss = model(b.control, b.control_total, b.case, b.case_total,
                             seq_emb, target_emb, b.modality, b.mode, b.has_seq, b.has_target, pert_mask,
                             mask_ratio=current_mask_ratio, beta_nll=current_beta, unknown_mask=unknown_mask, dose=b.dose)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(trainable_params, 5.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        total_epoch_loss += loss.item()

        if writer:
            writer.add_scalar('loss/train', loss.item(), step)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], step)
            writer.add_scalar('grad_norm', grad_norm.item(), step)
            writer.add_scalar('mask_ratio', current_mask_ratio, step)
            writer.add_scalar('beta_nll', current_beta, step)
            for k, v in components.items():
                writer.add_scalar(f'loss/{k}', v.item(), step)

        if step % 100 == 0:
            print(f'Step {step} | Loss: {loss.item():.5f} | LR: {scheduler.get_last_lr()[0]:.2e} | Beta: {current_beta:.3f}')

        if step > 0 and (step + 1) % steps_per_epoch == 0:
            epoch = (step + 1) // steps_per_epoch
            avg_loss = total_epoch_loss / steps_per_epoch
            print(f'=== Epoch {epoch} Done. Avg Loss: {avg_loss:.5f} | Mask: {current_mask_ratio:.3f} | Beta: {current_beta:.3f} ===')
            total_epoch_loss = 0

        if last_step:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, checkpoint_dir / f'biojepa_{VERSION}_ac_final.pt')

    if writer:
        writer.close()

    return {'loss_history': loss_history, 'final_loss': loss_history[-1] if loss_history else None}


def train_linear_decoder(model, train_loader, val_loader, seq_banks, target_bank, model_cfg, device, checkpoint_dir, cfg, use_amp=False, use_fused_optimizer=False, log_dir='default'):
    reset_seed()
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if log_dir == 'default':
        log_dir = checkpoint_dir / 'training_logs'
    writer = _make_writer(log_dir, 'decoder')

    use_autocast = use_amp and device.type == 'cuda'
    fused = use_fused_optimizer and torch.cuda.is_available()

    decoder_config = BenchmarkDecoderConfig(embed_dim=model_cfg.embed_dim)
    decoder = BenchmarkDecoder(decoder_config).to(device)

    steps_per_epoch = train_loader.total_samples // train_loader.batch_size
    if cfg.epochs is not None:
        max_steps = cfg.epochs * steps_per_epoch
    elif cfg.n_steps is not None:
        max_steps = cfg.n_steps
    else:
        raise ValueError('Either epochs or n_steps must be specified')
    print(f'Decoder training: {train_loader.total_samples} samples, {steps_per_epoch} steps/epoch, {max_steps} total steps')

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=cfg.lr, fused=fused)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=max_steps, pct_start=0.05)

    loss_fn = nn.MSELoss()
    loss_history = []

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    decoder.train()

    for step in range(max_steps):
        last_step = (step == max_steps - 1)

        if step % 500 == 0 or last_step:
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

                    unknown_mask = ~b.gene_mask

                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
                        z_context = model.student(b.control, b.control_total, mask_idx=None, unknown_mask=unknown_mask)
                        action_latents = model.composer(seq_emb, target_emb, b.modality, b.mode, b.has_seq, b.has_target, pert_mask, dose=b.dose)
                        target_indices = torch.arange(N, device=device).expand(B, N)
                        z_pred_mu, _ = model.predictor(z_context, action_latents, target_indices)
                        pred_delta = decoder(z_pred_mu) - decoder(z_context)
                        real_delta = b.case - b.control
                        val_loss = loss_fn(pred_delta[b.gene_mask], real_delta[b.gene_mask])
                    val_loss_accum += val_loss.item()
                avg_val_loss = val_loss_accum / val_loss_steps
                print(f'Decoder Step {step} | val loss: {avg_val_loss:.4f}')
                if writer:
                    writer.add_scalar('loss/val', avg_val_loss, step)
            decoder.train()

        b = train_loader.next_batch()
        B, N = b.control.shape

        seq_emb = get_seq_embeddings(b.seq_idx, b.modality, seq_banks)
        target_emb = get_target_embeddings(b.target_idx, target_bank)
        pert_mask = torch.arange(b.seq_idx.shape[1], device=device).unsqueeze(0) < b.n_perts.unsqueeze(1)
        unknown_mask = ~b.gene_mask

        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
            z_context = model.student(b.control, b.control_total, mask_idx=None, unknown_mask=unknown_mask)
            action_latents = model.composer(seq_emb, target_emb, b.modality, b.mode, b.has_seq, b.has_target, pert_mask, dose=b.dose)
            target_indices = torch.arange(N, device=device).expand(B, N)
            z_pred_mu, _ = model.predictor(z_context, action_latents, target_indices)

        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_autocast):
            pred_delta = decoder(z_pred_mu) - decoder(z_context)
            real_delta = b.case - b.control
            loss = loss_fn(pred_delta[b.gene_mask], real_delta[b.gene_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        if writer:
            writer.add_scalar('loss/train', loss.item(), step)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], step)

        if step % 100 == 0:
            print(f'Decoder Step {step} | Loss: {loss.item():.5f}')

        if last_step:
            torch.save({
                'model': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, checkpoint_dir / f'biojepa_{VERSION}_decoder_final.pt')

    if writer:
        writer.close()

    return decoder, {'loss_history': loss_history, 'final_loss': loss_history[-1] if loss_history else None}
