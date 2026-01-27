import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from biojepa_v0_6 import BioJepa, BioJepaConfig
from evals.linear_expression_decoder import BenchmarkDecoder, BenchmarkDecoderConfig
from config_v0_6 import PretrainConfig, AlignmentConfig, FullTrainingConfig, DecoderConfig, DataConfig, MODALITY_TO_ID, MODE_TO_ID


def create_model(model_cfg: BioJepaConfig, device) -> BioJepa:
    model = BioJepa(model_cfg).to(device)
    return model


def load_feature_banks(data_cfg: DataConfig, device):
    input_bank = torch.from_numpy(np.load(data_cfg.input_bank_path)).float().to(device)
    anchor_bank = torch.from_numpy(np.load(data_cfg.anchor_bank_path)).float().to(device)
    print(f'Banks Loaded. Input(DNA): {input_bank.shape}, Anchor(Prot): {anchor_bank.shape}')
    return input_bank, anchor_bank


def run_pretraining(model, train_loader, val_loader, cfg: PretrainConfig, device, checkpoint_dir, model_cfg: BioJepaConfig) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = train_loader.total_samples // cfg.batch_size
    max_steps = cfg.epochs * steps_per_epoch
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
                val_loss_steps = 10
                for _ in range(val_loss_steps):
                    x_val, total_val = val_loader.next_batch()
                    val_loss = model.forward_pretrain(x_val, total_val)
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

        x, total = train_loader.next_batch()
        optimizer.zero_grad()
        loss = model.forward_pretrain(x, total)
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


def run_alignment(model, train_loader, val_loader, input_bank, anchor_bank, cfg: AlignmentConfig, device, checkpoint_dir) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = train_loader.total_samples // cfg.batch_size
    max_steps = cfg.epochs * steps_per_epoch
    print(f'Alignment: {train_loader.total_samples} samples, {steps_per_epoch} steps/epoch, {max_steps} total steps')

    for p in model.parameters():
        p.requires_grad = False
    for p in model.composer.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.composer.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=max_steps, pct_start=0.05
    )

    loss_history = []
    total_epoch_loss = 0
    model.train()

    for step in range(max_steps):
        last_step = (step == max_steps - 1)

        if step % 500 == 0 or last_step:
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 10
                for _ in range(val_loss_steps):
                    inp_idx, anc_idx, inp_mod, inp_mode = val_loader.next_batch()
                    inp_feats = input_bank[inp_idx]
                    anc_feats = anchor_bank[anc_idx]
                    B = inp_idx.shape[0]
                    anc_mod = torch.full((B,), MODALITY_TO_ID['protein'], device=device, dtype=torch.long)
                    anc_mode = torch.full((B,), MODE_TO_ID['control'], device=device, dtype=torch.long)
                    val_loss = model.forward_alignment(
                        anchor_feats=anc_feats, anchor_mod=anc_mod, anchor_mode=anc_mode,
                        positive_feats=inp_feats, positive_mod=inp_mod, positive_mode=inp_mode
                    )
                    val_loss_accum += val_loss.item()
                avg_val_loss = val_loss_accum / val_loss_steps
                print(f'Step {step} | val loss: {avg_val_loss:.4f}')
            model.train()

        # commenting since we don't want to be noisy. 
        # if step > 0 and (step + 1) % steps_per_epoch == 0 and not last_step:
        #     epoch = (step + 1) // steps_per_epoch
        #     torch.save({
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'step': step
        #     }, checkpoint_dir / f'biojepa_v0_6_align_epoch_{epoch}.pt')

        inp_idx, anc_idx, inp_mod, inp_mode = train_loader.next_batch()
        inp_feats = input_bank[inp_idx]
        anc_feats = anchor_bank[anc_idx]
        B = inp_idx.shape[0]
        anc_mod = torch.full((B,), MODALITY_TO_ID['protein'], device=device, dtype=torch.long)
        anc_mode = torch.full((B,), MODE_TO_ID['control'], device=device, dtype=torch.long)

        optimizer.zero_grad()
        loss = model.forward_alignment(
            anchor_feats=anc_feats, anchor_mod=anc_mod, anchor_mode=anc_mode,
            positive_feats=inp_feats, positive_mod=inp_mod, positive_mode=inp_mode
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        #total_epoch_loss += loss.item()

        if step % 100 == 0:
            print(f'Step {step} | Loss: {loss.item():.5f} | LR: {scheduler.get_last_lr()[0]:.2e}')

        # if step > 0 and (step + 1) % steps_per_epoch == 0:
        #     avg_loss = total_epoch_loss / steps_per_epoch
        #     print(f'=== Epoch {(step + 1) // steps_per_epoch} Done. Avg Loss: {avg_loss:.5f} ===')
        #     total_epoch_loss = 0

        if last_step:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step
            }, checkpoint_dir / f'biojepa_v0_6_align_final.pt')

    return {'loss_history': loss_history, 'final_loss': loss_history[-1] if loss_history else None}


def run_full_training(model, train_loader, val_loader, input_bank, cfg: FullTrainingConfig, device, checkpoint_dir) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.freeze_encoders()
    for p in model.predictor.parameters():
        p.requires_grad = True
    for p in model.composer.parameters():
        p.requires_grad = True

    steps_per_epoch = train_loader.total_samples // cfg.batch_size
    max_steps = cfg.epochs * steps_per_epoch
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
                val_loss_steps = 10
                for _ in range(val_loss_steps):
                    xc, xct, xt, xtt, p_idx, p_mod, p_mode = val_loader.next_batch()
                    p_feats = input_bank[p_idx]
                    val_loss = model(xc, xct, xt, xtt, p_feats, p_mod, p_mode)
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

        xc, xct, xt, xtt, p_idx, p_mod, p_mode = train_loader.next_batch()
        p_feats = input_bank[p_idx]

        optimizer.zero_grad()
        loss = model(xc, xct, xt, xtt, p_feats, p_mod, p_mode)
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


def train_linear_decoder(model, train_loader, val_loader, input_bank, model_cfg: BioJepaConfig, device, checkpoint_dir, cfg: DecoderConfig) -> BenchmarkDecoder:
    '''Train linear decoder on action-conditioned predictions.

    Uses the full prediction pipeline: student encoder -> composer -> predictor.
    Decoder learns to map latent deltas (z_pred - z_context) to expression deltas (xt - xc).
    '''
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    decoder_config = BenchmarkDecoderConfig(embed_dim=model_cfg.embed_dim)
    decoder = BenchmarkDecoder(decoder_config).to(device)

    steps_per_epoch = train_loader.total_samples // train_loader.batch_size
    max_steps = cfg.epochs * steps_per_epoch

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
                val_loss_steps = 10
                for _ in range(val_loss_steps):
                    xc, xct, xt, xtt, p_idx, p_mod, p_mode = val_loader.next_batch()
                    p_feats = input_bank[p_idx]
                    B, N = xc.shape

                    z_context = model.student(xc, xct, mask_idx=None)
                    action_latents = model.composer(p_feats, p_mod, p_mode)
                    target_indices = torch.arange(N, device=device).expand(B, N)
                    z_pred_mu, _ = model.predictor(z_context, action_latents, target_indices)

                    pred_delta = decoder(z_pred_mu) - decoder(z_context)
                    real_delta = xt - xc
                    val_loss = loss_fn(pred_delta, real_delta)
                    val_loss_accum += val_loss.item()
                avg_val_loss = val_loss_accum / val_loss_steps
                print(f'Decoder Step {step} | val loss: {avg_val_loss:.4f}')
            decoder.train()

        xc, xct, xt, xtt, p_idx, p_mod, p_mode = train_loader.next_batch()
        p_feats = input_bank[p_idx]
        B, N = xc.shape

        with torch.no_grad():
            z_context = model.student(xc, xct, mask_idx=None)
            action_latents = model.composer(p_feats, p_mod, p_mode)
            target_indices = torch.arange(N, device=device).expand(B, N)
            z_pred_mu, _ = model.predictor(z_context, action_latents, target_indices)

        optimizer.zero_grad()
        pred_delta = decoder(z_pred_mu) - decoder(z_context)
        real_delta = xt - xc
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

    return decoder
