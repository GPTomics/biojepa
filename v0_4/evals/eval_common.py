import sys
sys.path.insert(0, '..')

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json
from datetime import datetime

import biojepa_ac_v0_4 as model
from bio_dataloader import TrainingLoader
from linear_expression_decoder import BenchmarkDecoder, BenchmarkDecoderConfig

DEFAULT_CONFIG = {
    'batch_size': 32,
    'n_genes': 5000,
    'n_layers': 2,
    'n_heads': 2,
    'n_embd': 8,
    'pert_latent_dim': 320,
    'pert_mode_dim': 64,
    'test_total_examples': 38829
}


def get_device():
    device = 'cpu'
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
        device = 'cuda'
    print(f'using {device}')
    return device


def get_paths(data_root='/Users/djemec/data/jepa/v0_4'):
    data_dir = Path(data_root)
    return {
        'data_dir': data_dir,
        'train_dir': data_dir / 'training',
        'checkpoint_dir': data_dir / 'checkpoints',
        'pert_dir': data_dir / 'pert_embd'
    }


def load_perturbation_bank(paths, device):
    input_bank = torch.from_numpy(np.load(paths['pert_dir'] / 'input_embeddings_dna.npy')).float().to(device)
    print(f'Input Bank (DNA): {input_bank.shape}')
    return input_bank


def load_biojepa_model(paths, device, config=None):
    config = config or DEFAULT_CONFIG
    torch.set_float32_matmul_precision('high')

    model_config = model.BioJepaConfig(
        num_genes=config['n_genes'],
        n_layer=config['n_layers'],
        heads=config['n_heads'],
        embed_dim=config['n_embd'],
        n_pre_layer=config['n_layers'],
        pert_latent_dim=config['pert_latent_dim'],
        pert_mode_dim=config['pert_mode_dim']
    )
    biojepa = model.BioJepa(model_config).to(device)

    checkpoint_path = paths['checkpoint_dir'] / 'bio_jepa_ckpt_31769_final.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    keys = biojepa.load_state_dict(checkpoint['model'])
    print(keys)

    biojepa.freeze_encoders()
    biojepa.eval()
    for param in biojepa.parameters():
        param.requires_grad = False

    return biojepa


def load_decoder(paths, device, config=None):
    config = config or DEFAULT_CONFIG
    decoder_config = BenchmarkDecoderConfig(embed_dim=config['n_embd'])
    decoder = BenchmarkDecoder(decoder_config).to(device)

    decoder_checkpoint_path = paths['checkpoint_dir'] / 'linear_decoder_ckpt_15884_final.pt'
    decoder_checkpoint = torch.load(decoder_checkpoint_path, map_location=device)
    decoder.load_state_dict(decoder_checkpoint['model'])
    decoder.eval()
    print('Decoder loaded')

    return decoder


def load_test_data(paths, device, config=None):
    config = config or DEFAULT_CONFIG
    test_loader = TrainingLoader(
        batch_size=config['batch_size'],
        split='test',
        data_dir=paths['train_dir'],
        device=device
    )
    test_steps = config['test_total_examples'] // config['batch_size']
    return test_loader, test_steps


class EvalContext:
    '''Holds all shared state for evaluations.'''

    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.device = get_device()
        self.paths = get_paths()
        self.input_bank = load_perturbation_bank(self.paths, self.device)
        self.biojepa = load_biojepa_model(self.paths, self.device, self.config)
        self.decoder = load_decoder(self.paths, self.device, self.config)

        self._cached_inference = None

    def run_test_inference(self, force=False):
        '''Run inference on test set, caching results. Returns aggregated and per-sample data including uncertainty.'''
        if self._cached_inference is not None and not force:
            print('Using cached inference results')
            return self._cached_inference

        test_loader, test_steps = load_test_data(self.paths, self.device, self.config)
        from scipy.stats import pearsonr

        bulk_pred_deltas = defaultdict(list)
        bulk_real_deltas = defaultdict(list)
        bulk_pred_abs = defaultdict(list)
        bulk_real_abs = defaultdict(list)

        sample_pred_deltas = []
        sample_real_deltas = []
        sample_logvars = []
        sample_pert_ids = []
        sample_mses = []
        sample_correlations = []

        N = self.config['n_genes']

        for step in tqdm(range(test_steps), desc='Running inference'):
            cont_x, cont_tot, case_x, case_tot, p_idx, p_mod, p_mode = test_loader.next_batch()
            p_feats = self.input_bank[p_idx]
            B = cont_x.shape[0]

            with torch.no_grad():
                z_context = self.biojepa.student(cont_x, cont_tot, mask_idx=None)
                action_latents = self.biojepa.composer(p_feats, p_mod, p_mode)
                target_indices = torch.arange(N, device=self.device).expand(B, N)
                z_pred_mu, z_pred_logvar = self.biojepa.predictor(z_context, action_latents, target_indices)

                pred_delta = self.decoder(z_pred_mu) - self.decoder(z_context)
                real_delta = case_x - cont_x
                pred_abs = torch.clamp(cont_x + pred_delta, min=0.0)

            pred_delta_np = pred_delta.cpu().numpy()
            real_delta_np = real_delta.cpu().numpy()
            pred_abs_np = pred_abs.cpu().numpy()
            real_abs_np = case_x.cpu().numpy()
            logvar_np = z_pred_logvar.mean(dim=-1).cpu().numpy()
            p_idx_np = p_idx.cpu().numpy().flatten()

            sample_pred_deltas.append(pred_delta_np)
            sample_real_deltas.append(real_delta_np)
            sample_logvars.append(logvar_np)
            sample_pert_ids.append(p_idx_np)

            for i in range(B):
                pid = p_idx_np[i]
                bulk_pred_deltas[pid].append(pred_delta_np[i])
                bulk_real_deltas[pid].append(real_delta_np[i])
                bulk_pred_abs[pid].append(pred_abs_np[i])
                bulk_real_abs[pid].append(real_abs_np[i])

                sample_mses.append(np.mean((pred_delta_np[i] - real_delta_np[i])**2))

                top_20_idx = np.argsort(np.abs(real_delta_np[i]))[-20:]
                p_top = pred_delta_np[i][top_20_idx]
                t_top = real_delta_np[i][top_20_idx]
                if np.std(p_top) > 1e-9 and np.std(t_top) > 1e-9:
                    corr, _ = pearsonr(p_top, t_top)
                    sample_correlations.append(0.0 if np.isnan(corr) else corr)
                else:
                    sample_correlations.append(0.0)

        pert_ids = list(bulk_pred_deltas.keys())

        self._cached_inference = {
            'pert_ids': pert_ids,
            'mean_pred_deltas': {pid: np.mean(np.stack(bulk_pred_deltas[pid]), axis=0) for pid in pert_ids},
            'mean_real_deltas': {pid: np.mean(np.stack(bulk_real_deltas[pid]), axis=0) for pid in pert_ids},
            'mean_pred_abs': {pid: np.mean(np.stack(bulk_pred_abs[pid]), axis=0) for pid in pert_ids},
            'mean_real_abs': {pid: np.mean(np.stack(bulk_real_abs[pid]), axis=0) for pid in pert_ids},
            'sample_mses': np.array(sample_mses),
            'sample_correlations': np.array(sample_correlations),
            'sample_pred_deltas': np.concatenate(sample_pred_deltas, axis=0),
            'sample_real_deltas': np.concatenate(sample_real_deltas, axis=0),
            'sample_logvars': np.concatenate(sample_logvars, axis=0),
            'sample_pert_ids': np.concatenate(sample_pert_ids, axis=0)
        }

        print(f'Aggregated {len(pert_ids)} perturbations, {len(self._cached_inference["sample_mses"])} samples')
        return self._cached_inference


def update_eval_report(eval_name, results, report_path='eval_report.json'):
    '''Update the eval_report.json file with results from an evaluation.'''
    report_path = Path(report_path)

    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        report = {'version': 'v0.4', 'evals': {}}

    report['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    report['evals'][eval_name] = {
        'run_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        **results
    }

    report_path.write_text(json.dumps(report, indent=2))
    print(f'Updated {report_path}')
