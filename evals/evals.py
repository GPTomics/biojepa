'''BioJEPA v0.6 Evaluation Suite

Three entry points:
- run_pretraining_evals(ctx): batch_invariance, gene_embedding_pathways, essential_gene_prediction,
                              cell_type_probing, reconstruction, perturbation_detection,
                              embedding_consistency, latent_space_health
- run_full_model_evals(ctx): expression_prediction, gene_level_analysis, perturbation_retrieval,
                             uncertainty_calibration, action_vector_pathways, moa_matching
- run_alignment_evals(ctx): seq_to_target_retrieval, cross_modality_target_consistency,
                            seq_target_gap_analysis, paired_alignment_quality, mode_sensitivity,
                            fusion_quality, missing_data_robustness, multi_pert_alignment,
                            target_family_probing
'''

import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score, f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist

import biojepa_v0_6 as model
from dataloader_v0_6 import TrainingLoader
from .linear_expression_decoder import BenchmarkDecoder, BenchmarkDecoderConfig
from .pathway_utils import load_pathway_annotations, map_genes_to_pathways, compute_pathway_clustering_metrics
from .linear_classifier import train_linear_classifier
import torch.nn.functional as F
from config_v0_6 import MAX_SEQ_DIM


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


REQUIRED_CONFIG_KEYS = ['num_genes', 'embed_dim', 'n_layer', 'heads', 'batch_size']


class EvalContext:
    '''Unified context for running evaluations. All loading is lazy.'''

    def __init__(self, config, data_root, checkpoint_root, ref_dir):
        missing_keys = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
        if missing_keys:
            raise ValueError(f'Missing required config keys: {missing_keys}. Required: {REQUIRED_CONFIG_KEYS}')
        self.config = {'test_total_examples': 38829, 'pert_latent_dim': 320, 'pert_mode_dim': 64, **config}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.paths = self._get_paths(Path(data_root), Path(checkpoint_root), Path(ref_dir))
        print(f'Using {self.device}')

        self._biojepa = None
        self._decoder = None
        self._gene_embeddings = None
        self._gene_names = None
        self._test_inference = None
        self._alignment_pairs = None
        self._alignment_inference = None
        self._hgnc_gene_families = None
        self._pathway_annotations = None
        self._id_to_gene = None
        self._seq_banks = None
        self._target_bank = None

    @classmethod
    def from_trained_model(cls, biojepa_model, data_root, ref_dir, config, decoder=None):
        '''Create context from in-memory trained model (for notebook integration).'''
        ctx = cls(config=config, data_root=data_root, checkpoint_root=data_root, ref_dir=ref_dir)
        ctx._biojepa = biojepa_model
        ctx._biojepa.freeze_encoders()
        ctx._biojepa.eval()
        for param in ctx._biojepa.parameters():
            param.requires_grad = False
        if decoder is not None:
            ctx._decoder = decoder
            ctx._decoder.eval()
        return ctx

    def _get_paths(self, data_root, checkpoint_root, ref_dir):
        return {
            'data_dir': data_root,
            'train_dir': data_root / 'training',
            'checkpoint_dir': checkpoint_root / 'checkpoints',
            'pert_dir': data_root / 'pert_embd',
            'ref_dir': ref_dir
        }

    @property
    def biojepa(self):
        if self._biojepa is None:
            print('Loading BioJEPA model...')
            torch.set_float32_matmul_precision('high')
            model_config = model.BioJepaConfig(
                num_genes=self.config['num_genes'], n_layer=self.config['n_layer'], heads=self.config['heads'],
                embed_dim=self.config['embed_dim'], n_pre_layer=self.config['n_layer'],
                pert_latent_dim=self.config['pert_latent_dim'], pert_mode_dim=self.config['pert_mode_dim']
            )
            self._biojepa = model.BioJepa(model_config).to(self.device)
            checkpoint = torch.load(self.paths['checkpoint_dir'] / 'biojepa_v0_6_full_final.pt', map_location=self.device)
            print(self._biojepa.load_state_dict(checkpoint['model']))
            self._biojepa.freeze_encoders()
            self._biojepa.eval()
            for param in self._biojepa.parameters():
                param.requires_grad = False
        return self._biojepa

    @property
    def decoder(self):
        if self._decoder is None:
            print('Loading decoder...')
            decoder_config = BenchmarkDecoderConfig(embed_dim=self.config['embed_dim'])
            self._decoder = BenchmarkDecoder(decoder_config).to(self.device)
            checkpoint = torch.load(self.paths['checkpoint_dir'] / 'biojepa_v0_6_decoder_final.pt', map_location=self.device)
            self._decoder.load_state_dict(checkpoint['model'])
            self._decoder.eval()
        return self._decoder

    @property
    def gene_embeddings(self):
        if self._gene_embeddings is None:
            self._gene_embeddings = self.biojepa.student.gene_embeddings.detach().cpu().numpy()
        return self._gene_embeddings

    @property
    def gene_names(self):
        if self._gene_names is None:
            with open(self.paths['data_dir'] / 'gene_names.json') as f:
                self._gene_names = json.load(f)
        return self._gene_names

    @property
    def seq_banks(self):
        '''Load sequence embedding banks (DNA, chemical) for v0.6 dual-path alignment.'''
        if self._seq_banks is None:
            seq_banks_dir = self.paths['pert_dir'] / 'seq_banks'
            self._seq_banks = {}
            dna_path = seq_banks_dir / 'dna_embeddings.npy'
            if dna_path.exists():
                self._seq_banks['dna'] = torch.from_numpy(np.load(dna_path)).float().to(self.device)
                print(f'Loaded DNA seq bank: {self._seq_banks["dna"].shape}')
            chem_path = seq_banks_dir / 'chemical_embeddings.npy'
            if chem_path.exists():
                self._seq_banks['chemical'] = torch.from_numpy(np.load(chem_path)).float().to(self.device)
                print(f'Loaded chemical seq bank: {self._seq_banks["chemical"].shape}')
        return self._seq_banks

    @property
    def target_bank(self):
        '''Load protein target embedding bank for v0.6 dual-path alignment.'''
        if self._target_bank is None:
            target_path = self.paths['pert_dir'] / 'target_banks' / 'protein_targets.npy'
            if target_path.exists():
                self._target_bank = torch.from_numpy(np.load(target_path)).float().to(self.device)
                print(f'Loaded target bank: {self._target_bank.shape}')
        return self._target_bank

    @property
    def alignment_pairs(self):
        '''Load alignment pairs - supports both old (v0.5) and new (v0.6) formats.'''
        if self._alignment_pairs is None:
            new_path = self.paths['pert_dir'] / 'train' / 'align_train.npz'
            old_path = self.paths['pert_dir'] / 'train' / 'pert_pairs_crispri_train.npz'
            if new_path.exists():
                with np.load(new_path) as data:
                    self._alignment_pairs = {
                        'seq_idx': data['seq_idx'],
                        'target_idx': data['target_idx'],
                        'modality': data['modality'],
                        'mode': data['mode']
                    }
                print(f'Loaded {len(self._alignment_pairs["seq_idx"])} v0.6 alignment pairs')
            elif old_path.exists():
                with np.load(old_path) as data:
                    n_pairs = len(data['input_idx'])
                    self._alignment_pairs = {
                        'seq_idx': data['input_idx'],
                        'target_idx': data['anchor_idx'],
                        'modality': np.zeros(n_pairs, dtype=np.int64),
                        'mode': np.zeros(n_pairs, dtype=np.int64)
                    }
                print(f'Loaded {n_pairs} v0.5 alignment pairs (converted to v0.6 format)')
            else:
                raise FileNotFoundError(f'No alignment pairs found at {new_path} or {old_path}')
        return self._alignment_pairs

    @property
    def alignment_inference(self):
        '''Cached action vectors for alignment evals using v0.6 dual-path architecture.

        Uses encode_sequence_path() for sequences and encode_target_path() for targets.
        Supports DNA and chemical modalities for sequences, protein targets only.
        '''
        if self._alignment_inference is None:
            result = {}

            with torch.no_grad():
                if self.seq_banks and 'dna' in self.seq_banks:
                    dna_emb = self.seq_banks['dna']
                    n_dna = dna_emb.shape[0]
                    seq_emb = torch.zeros(n_dna, 1, MAX_SEQ_DIM, device=self.device)
                    padded_dna = F.pad(dna_emb, (0, MAX_SEQ_DIM - dna_emb.shape[-1])) if dna_emb.shape[-1] < MAX_SEQ_DIM else dna_emb
                    seq_emb[:, 0, :] = padded_dna
                    modality_ids = torch.zeros(n_dna, 1, dtype=torch.long, device=self.device)
                    mode_ids = torch.zeros(n_dna, 1, dtype=torch.long, device=self.device)
                    pert_mask = torch.ones(n_dna, 1, dtype=torch.bool, device=self.device)
                    dna_actions = self.biojepa.composer.encode_sequence_path(seq_emb, modality_ids, mode_ids, pert_mask)
                    dna_actions = dna_actions.squeeze(1)
                    result['dna_actions'] = dna_actions
                    result['dna_actions_norm'] = F.normalize(dna_actions, dim=1)
                    print(f'Encoded DNA sequences: {dna_actions.shape}')

                if self.seq_banks and 'chemical' in self.seq_banks:
                    chem_emb = self.seq_banks['chemical']
                    n_chem = chem_emb.shape[0]
                    seq_emb = torch.zeros(n_chem, 1, MAX_SEQ_DIM, device=self.device)
                    padded_chem = F.pad(chem_emb, (0, MAX_SEQ_DIM - chem_emb.shape[-1])) if chem_emb.shape[-1] < MAX_SEQ_DIM else chem_emb
                    seq_emb[:, 0, :] = padded_chem
                    modality_ids = torch.full((n_chem, 1), 2, dtype=torch.long, device=self.device)
                    mode_ids = torch.full((n_chem, 1), 4, dtype=torch.long, device=self.device)
                    pert_mask = torch.ones(n_chem, 1, dtype=torch.bool, device=self.device)
                    chem_actions = self.biojepa.composer.encode_sequence_path(seq_emb, modality_ids, mode_ids, pert_mask)
                    chem_actions = chem_actions.squeeze(1)
                    result['chem_actions'] = chem_actions
                    result['chem_actions_norm'] = F.normalize(chem_actions, dim=1)
                    print(f'Encoded chemical sequences: {chem_actions.shape}')

                if self.target_bank is not None:
                    target_emb = self.target_bank
                    n_targets = target_emb.shape[0]
                    target_emb_batched = target_emb.unsqueeze(1)
                    mode_ids = torch.zeros(n_targets, 1, dtype=torch.long, device=self.device)
                    pert_mask = torch.ones(n_targets, 1, dtype=torch.bool, device=self.device)
                    target_actions = self.biojepa.composer.encode_target_path(target_emb_batched, mode_ids, pert_mask)
                    target_actions = target_actions.squeeze(1)
                    result['target_actions'] = target_actions
                    result['target_actions_norm'] = F.normalize(target_actions, dim=1)
                    print(f'Encoded protein targets: {target_actions.shape}')

            self._alignment_inference = result
        return self._alignment_inference

    @property
    def hgnc_gene_families(self):
        if self._hgnc_gene_families is None:
            hgnc_path = self.paths['ref_dir'] / 'gene_family' / 'hgnc.tsv'
            gene_to_family, ensg_to_family = {}, {}
            with open(hgnc_path) as f:
                next(f)
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 16 and parts[15]:
                        symbol = parts[1].upper()
                        ensg = parts[10] if len(parts) > 10 else None
                        family = parts[15]
                        gene_to_family[symbol] = family
                        if ensg:
                            ensg_to_family[ensg] = family
            self._hgnc_gene_families = {'by_symbol': gene_to_family, 'by_ensg': ensg_to_family}
            print(f'Loaded {len(gene_to_family)} HGNC gene family annotations')
        return self._hgnc_gene_families

    @property
    def pathway_annotations(self):
        '''Cached pathway annotations for KEGG and Reactome.'''
        if self._pathway_annotations is None:
            self._pathway_annotations = load_pathway_annotations(['KEGG_2021_Human', 'Reactome_Pathways_2024'])
        return self._pathway_annotations

    @property
    def id_to_gene(self):
        '''Cached mapping from perturbation ID to gene symbol.'''
        if self._id_to_gene is None:
            with open(self.paths['pert_dir'] / 'input_to_id.json') as f:
                input_to_id = json.load(f)
            self._id_to_gene = {pid: key.split('_')[0].upper() for key, pid in input_to_id.items()}
        return self._id_to_gene

    @property
    def test_inference(self):
        if self._test_inference is None:
            self._test_inference = self._run_test_inference()
        return self._test_inference

    def _run_test_inference(self):
        '''Run inference on test set. Returns aggregated and per-sample data.'''
        test_loader = TrainingLoader(batch_size=self.config['batch_size'], split='test', data_dir=self.paths['train_dir'], device=self.device)
        test_steps = self.config['test_total_examples'] // self.config['batch_size']
        N = self.config['num_genes']

        bulk_pred_deltas, bulk_real_deltas = defaultdict(list), defaultdict(list)
        bulk_pred_abs, bulk_real_abs = defaultdict(list), defaultdict(list)
        bulk_control_states = defaultdict(list)
        sample_pred_deltas, sample_real_deltas, sample_logvars = [], [], []
        sample_pert_ids, sample_target_ids, sample_pert_mods = [], [], []
        sample_mses, sample_correlations = [], []

        for _ in tqdm(range(test_steps), desc='Running test inference'):
            batch = test_loader.next_batch()
            cont_x, cont_tot = batch.control, batch.control_total
            case_x, case_tot = batch.case, batch.case_total
            B = cont_x.shape[0]
            N_pert = batch.seq_idx.shape[1]

            pert_mask = torch.arange(N_pert, device=self.device).unsqueeze(0) < batch.n_perts.unsqueeze(1)
            seq_emb = get_seq_embeddings(batch.seq_idx, batch.modality, self.seq_banks)
            target_emb = get_target_embeddings(batch.target_idx, self.target_bank)

            with torch.no_grad():
                z_context = self.biojepa.student(cont_x, cont_tot, mask_idx=None)
                action_latents = self.biojepa.composer(
                    seq_emb, target_emb, batch.modality, batch.mode,
                    batch.has_seq, batch.has_target, pert_mask
                )
                target_indices = torch.arange(N, device=self.device).expand(B, N)
                z_pred_mu, z_pred_logvar = self.biojepa.predictor(z_context, action_latents, target_indices)
                pred_delta = self.decoder(z_pred_mu) - self.decoder(z_context)
                real_delta = case_x - cont_x
                pred_abs = torch.clamp(cont_x + pred_delta, min=0.0)

            pred_delta_np, real_delta_np = pred_delta.cpu().numpy(), real_delta.cpu().numpy()
            pred_abs_np, real_abs_np = pred_abs.cpu().numpy(), case_x.cpu().numpy()
            logvar_np = z_pred_logvar.mean(dim=-1).cpu().numpy()
            p_idx_np = batch.seq_idx[:, 0].cpu().numpy()
            p_target_np = batch.target_idx[:, 0].cpu().numpy()
            p_mod_np = batch.modality[:, 0].cpu().numpy()
            cont_x_np = cont_x.cpu().numpy()

            sample_pred_deltas.append(pred_delta_np)
            sample_real_deltas.append(real_delta_np)
            sample_logvars.append(logvar_np)
            sample_pert_ids.append(p_idx_np)
            sample_target_ids.append(p_target_np)
            sample_pert_mods.append(p_mod_np)

            for i in range(B):
                key = (int(p_idx_np[i]), int(p_target_np[i]), int(p_mod_np[i]))
                bulk_pred_deltas[key].append(pred_delta_np[i])
                bulk_real_deltas[key].append(real_delta_np[i])
                bulk_pred_abs[key].append(pred_abs_np[i])
                bulk_real_abs[key].append(real_abs_np[i])
                bulk_control_states[key].append(cont_x_np[i])

                sample_mses.append(np.mean((pred_delta_np[i] - real_delta_np[i])**2))
                top_20_idx = np.argsort(np.abs(real_delta_np[i]))[-20:]
                p_top, t_top = pred_delta_np[i][top_20_idx], real_delta_np[i][top_20_idx]
                if np.std(p_top) > 1e-9 and np.std(t_top) > 1e-9:
                    corr, _ = pearsonr(p_top, t_top)
                    sample_correlations.append(0.0 if np.isnan(corr) else corr)
                else:
                    sample_correlations.append(0.0)

        pert_keys = list(bulk_pred_deltas.keys())
        print(f'Aggregated {len(pert_keys)} perturbations, {len(sample_mses)} samples')

        return {
            'pert_keys': pert_keys,
            'mean_pred_deltas': {k: np.mean(np.stack(bulk_pred_deltas[k]), axis=0) for k in pert_keys},
            'mean_real_deltas': {k: np.mean(np.stack(bulk_real_deltas[k]), axis=0) for k in pert_keys},
            'mean_pred_abs': {k: np.mean(np.stack(bulk_pred_abs[k]), axis=0) for k in pert_keys},
            'mean_real_abs': {k: np.mean(np.stack(bulk_real_abs[k]), axis=0) for k in pert_keys},
            'mean_control_states': {k: np.mean(np.stack(bulk_control_states[k]), axis=0) for k in pert_keys},
            'sample_mses': np.array(sample_mses),
            'sample_correlations': np.array(sample_correlations),
            'sample_pred_deltas': np.concatenate(sample_pred_deltas, axis=0),
            'sample_real_deltas': np.concatenate(sample_real_deltas, axis=0),
            'sample_logvars': np.concatenate(sample_logvars, axis=0),
            'sample_pert_ids': np.concatenate(sample_pert_ids, axis=0),
            'sample_target_ids': np.concatenate(sample_target_ids, axis=0),
            'sample_pert_mods': np.concatenate(sample_pert_mods, axis=0),
        }


def save_report(results, output_path='eval_report.json'):
    '''Save evaluation results to JSON report.'''
    report_path = Path(output_path)
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        report = {'version': 'v0.6', 'evals': {}}

    run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    report['last_updated'] = run_timestamp
    for name, res in results.items():
        report['evals'][name] = {'run_date': run_timestamp, **res}

    report_path.write_text(json.dumps(report, indent=2))
    print(f'Saved report to {report_path}')


# =============================================================================
# ENTRY POINTS
# =============================================================================

def run_pretraining_evals(ctx):
    '''Run pretraining (encoder-only) evaluations.'''
    return {
        'batch_invariance': _batch_invariance(ctx),
        'gene_embedding_pathways': _gene_embedding_pathways(ctx),
        'essential_gene_prediction': _essential_gene_prediction(ctx),
        'cell_type_probing': _cell_type_probing(ctx),
        'reconstruction': _reconstruction(ctx),
        'perturbation_detection': _perturbation_detection(ctx),
        'embedding_consistency': _embedding_consistency(ctx),
        'latent_space_health': _latent_space_health(ctx),
    }


def run_full_model_evals(ctx):
    '''Run full model evaluations.'''
    return {
        'expression_prediction': _expression_prediction(ctx),
        'gene_level_analysis': _gene_level_analysis(ctx),
        'perturbation_retrieval': _perturbation_retrieval(ctx),
        'uncertainty_calibration': _uncertainty_calibration(ctx),
        'action_vector_pathways': _action_vector_pathways(ctx),
        'moa_matching': _moa_matching(ctx),
    }


def run_alignment_evals(ctx):
    '''Run alignment stage evaluations (9 evals for v0.6 dual-path architecture).'''
    return {
        'seq_to_target_retrieval': _seq_to_target_retrieval(ctx),
        'cross_modality_target_consistency': _cross_modality_target_consistency(ctx),
        'seq_target_gap_analysis': _seq_target_gap_analysis(ctx),
        'paired_alignment_quality': _paired_alignment_quality(ctx),
        'mode_sensitivity': _mode_sensitivity(ctx),
        'fusion_quality': _fusion_quality(ctx),
        'missing_data_robustness': _missing_data_robustness(ctx),
        'multi_pert_alignment': _multi_pert_alignment(ctx),
        'target_family_probing': _target_family_probing(ctx),
    }


# =============================================================================
# PRETRAINING EVALS
# =============================================================================

def _batch_invariance(ctx):
    '''Are representations confounded by batch effects?'''
    test_loader = TrainingLoader(batch_size=ctx.config['batch_size'], split='test', data_dir=ctx.paths['train_dir'], device=ctx.device)
    test_steps = ctx.config['test_total_examples'] // ctx.config['batch_size']

    all_emb, all_batch, all_pert = [], [], []
    with torch.no_grad():
        for _ in tqdm(range(test_steps), desc='batch_invariance: Extracting embeddings'):
            batch = test_loader.next_batch()
            cont_x, cont_tot = batch.control, batch.control_total
            all_emb.append(ctx.biojepa.student(cont_x, cont_tot, mask_idx=None).mean(dim=1).cpu().numpy())
            all_batch.append(batch.batch_id.cpu().numpy())
            all_pert.append(batch.seq_idx[:, 0].cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0)
    batch_ids = np.concatenate(all_batch, axis=0).flatten()
    pert_ids = np.concatenate(all_pert, axis=0).flatten()

    batch_map = {b: i for i, b in enumerate(sorted(np.unique(batch_ids)))}
    pert_map = {p: i for i, p in enumerate(sorted(np.unique(pert_ids)))}
    batch_labels = np.array([batch_map[b] for b in batch_ids])
    pert_labels = np.array([pert_map[p] for p in pert_ids])

    n_batch, n_pert = len(batch_map), len(pert_map)
    train_idx, val_idx = train_test_split(np.arange(len(embeddings)), test_size=0.2, random_state=42)

    print('Training batch classifier...')
    _, _, batch_acc = train_linear_classifier(embeddings[train_idx], batch_labels[train_idx], embeddings[val_idx], batch_labels[val_idx], n_batch, ctx.device, epochs=100)
    print('Training perturbation classifier...')
    _, _, pert_acc = train_linear_classifier(embeddings[train_idx], pert_labels[train_idx], embeddings[val_idx], pert_labels[val_idx], n_pert, ctx.device, epochs=100)

    batch_chance, pert_chance = 1.0 / n_batch, 1.0 / n_pert
    print(f'batch_invariance: Batch={batch_acc:.4f} ({batch_acc/batch_chance:.1f}x), Pert={pert_acc:.4f} ({pert_acc/pert_chance:.1f}x)')

    return {
        'config': {'samples': len(embeddings), 'embedding_dim': int(embeddings.shape[1]), 'num_batches': n_batch, 'num_perturbations': n_pert},
        'batch_classifier': {'accuracy': float(batch_acc), 'chance': float(batch_chance), 'above_chance_ratio': float(batch_acc / batch_chance)},
        'perturbation_classifier': {'accuracy': float(pert_acc), 'chance': float(pert_chance), 'above_chance_ratio': float(pert_acc / pert_chance)},
        'invariance_ratio': float(pert_acc / batch_acc) if batch_acc > 0 else 0.0
    }


def _gene_embedding_pathways(ctx):
    '''Do genes in same pathway cluster in learned embeddings?'''
    pathway_libs = ctx.pathway_annotations

    gene_labels_kegg, pathway_to_genes_kegg = map_genes_to_pathways(ctx.gene_names, pathway_libs['KEGG_2021_Human'], min_pathway_size=15, max_pathway_size=300)
    kegg_idx = list(set(idx for indices in pathway_to_genes_kegg.values() for idx in indices))
    kegg_labels = [gene_labels_kegg[ctx.gene_names[i].upper()] for i in kegg_idx]
    kegg_metrics = compute_pathway_clustering_metrics(ctx.gene_embeddings[kegg_idx], kegg_labels, min_samples_per_class=10)

    gene_labels_react, pathway_to_genes_react = map_genes_to_pathways(ctx.gene_names, pathway_libs['Reactome_Pathways_2024'], min_pathway_size=15, max_pathway_size=300)
    react_idx = list(set(idx for indices in pathway_to_genes_react.values() for idx in indices))
    react_labels = [gene_labels_react[ctx.gene_names[i].upper()] for i in react_idx]
    react_metrics = compute_pathway_clustering_metrics(ctx.gene_embeddings[react_idx], react_labels, min_samples_per_class=10)

    print(f'gene_embedding_pathways: KEGG sil={kegg_metrics["silhouette_score"]:.4f}, Reactome sil={react_metrics["silhouette_score"]:.4f}')
    return {'config': {'n_genes': len(ctx.gene_names)}, 'kegg': kegg_metrics, 'reactome': react_metrics}


def _essential_gene_prediction(ctx):
    '''Do gene embeddings encode functional importance?'''
    depmap_file = ctx.paths['ref_dir'] / 'depmap' / 'CRISPRGeneEffect.csv'
    if not depmap_file.exists():
        return {'error': f'DepMap file not found: {depmap_file}'}

    crispr_df = pd.read_csv(depmap_file, index_col=0)
    K562_ID = 'ACH-000551'
    if K562_ID not in crispr_df.index:
        return {'error': 'K562 not found in DepMap data'}

    k562_scores = crispr_df.loc[K562_ID]
    gene_to_score = {col.split(' ')[0].upper(): k562_scores[col] for col in k562_scores.index}

    matched_idx, matched_scores = [], []
    for idx, gene in enumerate(ctx.gene_names):
        score = gene_to_score.get(gene.upper())
        if score is not None and not np.isnan(score):
            matched_idx.append(idx)
            matched_scores.append(score)

    X, y = ctx.gene_embeddings[matched_idx], np.array(matched_scores)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    class Probe(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.fc = nn.Linear(d, 1)
        def forward(self, x):
            return self.fc(x).squeeze(-1)

    probe = Probe(X_train.shape[1]).to(ctx.device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    X_t, y_t = torch.from_numpy(X_train).float().to(ctx.device), torch.from_numpy(y_train).float().to(ctx.device)

    for _ in range(500):
        opt.zero_grad()
        nn.MSELoss()(probe(X_t), y_t).backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        opt.step()

    probe.eval()
    with torch.no_grad():
        y_pred_test = probe(torch.from_numpy(X_test).float().to(ctx.device)).cpu().numpy()

    pearson_test, _ = pearsonr(y_test, y_pred_test)
    spearman_test, _ = spearmanr(y_test, y_pred_test)

    THRESH = -0.5
    y_test_bin = (y_test < THRESH).astype(int)
    auroc_test = roc_auc_score(y_test_bin, -y_pred_test)

    print(f'essential_gene_prediction: Pearson={pearson_test:.4f}, AUROC={auroc_test:.4f}')

    return {
        'config': {'matched_genes': len(matched_idx), 'train_genes': len(X_train), 'test_genes': len(X_test)},
        'regression': {'pearson_test': float(pearson_test), 'spearman_test': float(spearman_test)},
        'classification': {'auroc_test': float(auroc_test), 'n_essential_test': int(y_test_bin.sum()), 'n_non_essential_test': int((~y_test_bin.astype(bool)).sum())}
    }


def _cell_type_probing(ctx):
    '''Can cell type be predicted from cell embeddings?'''
    test_loader = TrainingLoader(batch_size=ctx.config['batch_size'], split='test', data_dir=ctx.paths['train_dir'], device=ctx.device)
    test_steps = ctx.config['test_total_examples'] // ctx.config['batch_size']

    all_emb, all_cell_type, all_batch_id = [], [], []
    with torch.no_grad():
        for _ in tqdm(range(test_steps), desc='cell_type_probing: Extracting embeddings'):
            batch = test_loader.next_batch()
            cont_x, cont_tot = batch.control, batch.control_total
            emb = ctx.biojepa.student(cont_x, cont_tot, mask_idx=None).mean(dim=1).cpu().numpy()
            all_emb.append(emb)
            all_cell_type.append(batch.cell_type.cpu().numpy())
            all_batch_id.append(batch.batch_id.cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0)
    cell_types = np.concatenate(all_cell_type, axis=0).flatten()
    batch_ids = np.concatenate(all_batch_id, axis=0).flatten()

    unique_types = sorted(np.unique(cell_types))
    valid_cell_types = []
    for ct in unique_types:
        ct_mask = cell_types == ct
        ct_batches = batch_ids[ct_mask]
        if len(np.unique(ct_batches)) > 1:
            valid_cell_types.append(ct)

    if len(valid_cell_types) < 2:
        return {'error': 'Not enough cell types with batch variation for meaningful probing',
                'config': {'total_cell_types': len(unique_types), 'valid_cell_types': len(valid_cell_types)}}

    valid_mask = np.isin(cell_types, valid_cell_types)
    embeddings = embeddings[valid_mask]
    cell_types_filtered = cell_types[valid_mask]

    type_map = {t: i for i, t in enumerate(valid_cell_types)}
    labels = np.array([type_map[t] for t in cell_types_filtered])
    n_classes = len(type_map)

    train_idx, val_idx = train_test_split(np.arange(len(embeddings)), test_size=0.2, random_state=42, stratify=labels)

    print('Training cell type classifier...')
    _, val_preds, val_acc = train_linear_classifier(embeddings[train_idx], labels[train_idx], embeddings[val_idx], labels[val_idx], n_classes, ctx.device, epochs=100)

    macro_f1 = f1_score(labels[val_idx], val_preds, average='macro')
    chance = 1.0 / n_classes

    print(f'cell_type_probing: Accuracy={val_acc:.4f} ({val_acc/chance:.1f}x chance), Macro F1={macro_f1:.4f}')

    return {
        'config': {'samples': len(embeddings), 'embedding_dim': int(embeddings.shape[1]), 'num_cell_types': n_classes, 'filtered_from': len(unique_types)},
        'metrics': {'accuracy': float(val_acc), 'macro_f1': float(macro_f1), 'chance': float(chance), 'above_chance_ratio': float(val_acc / chance)}
    }


def _reconstruction(ctx):
    '''Can gene expression be reconstructed from embeddings?'''
    test_loader = TrainingLoader(batch_size=ctx.config['batch_size'], split='test', data_dir=ctx.paths['train_dir'], device=ctx.device)
    test_steps = ctx.config['test_total_examples'] // ctx.config['batch_size']
    n_genes = ctx.config['num_genes']

    all_emb, all_expr = [], []
    with torch.no_grad():
        for _ in tqdm(range(test_steps), desc='reconstruction: Extracting embeddings'):
            batch = test_loader.next_batch()
            cont_x, cont_tot = batch.control, batch.control_total
            emb = ctx.biojepa.student(cont_x, cont_tot, mask_idx=None).cpu().numpy()
            all_emb.append(emb)
            all_expr.append(cont_x.cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0)
    expressions = np.concatenate(all_expr, axis=0)

    n_samples = embeddings.shape[0]
    gene_perm = np.random.RandomState(42).permutation(n_genes)
    n_train_genes = int(0.8 * n_genes)
    train_genes, test_genes = gene_perm[:n_train_genes], gene_perm[n_train_genes:]

    X_train = embeddings[:, train_genes, :].reshape(-1, embeddings.shape[-1])
    y_train = expressions[:, train_genes].reshape(-1)
    X_test = embeddings[:, test_genes, :].reshape(-1, embeddings.shape[-1])
    y_test = expressions[:, test_genes].reshape(-1)

    class ReconMLP(nn.Module):
        def __init__(self, d_in, d_hidden=64):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, 1))
        def forward(self, x):
            return self.net(x).squeeze(-1)

    mlp = ReconMLP(X_train.shape[1]).to(ctx.device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    train_subset = np.random.RandomState(42).choice(len(X_train), min(100000, len(X_train)), replace=False)
    X_t = torch.from_numpy(X_train[train_subset]).float().to(ctx.device)
    y_t = torch.from_numpy(y_train[train_subset]).float().to(ctx.device)

    print('Training reconstruction MLP...')
    for _ in range(200):
        opt.zero_grad()
        nn.MSELoss()(mlp(X_t), y_t).backward()
        opt.step()

    mlp.eval()
    with torch.no_grad():
        test_subset = np.random.RandomState(42).choice(len(X_test), min(50000, len(X_test)), replace=False)
        X_test_t = torch.from_numpy(X_test[test_subset]).float().to(ctx.device)
        y_pred = mlp(X_test_t).cpu().numpy()
        y_true = y_test[test_subset]

    mse = np.mean((y_pred - y_true)**2)
    pearson_r, _ = pearsonr(y_pred, y_true)

    print(f'reconstruction: MSE={mse:.4f}, Pearson R={pearson_r:.4f}')

    return {
        'config': {'samples': n_samples, 'train_genes': len(train_genes), 'test_genes': len(test_genes), 'embedding_dim': int(embeddings.shape[-1])},
        'metrics': {'reconstruction_mse': float(mse), 'pearson_r': float(pearson_r), 'pearson_r_squared': float(pearson_r**2)}
    }


def _perturbation_detection(ctx):
    '''Can we distinguish perturbed cells from control cells?'''
    test_loader = TrainingLoader(batch_size=ctx.config['batch_size'], split='test', data_dir=ctx.paths['train_dir'], device=ctx.device)
    test_steps = ctx.config['test_total_examples'] // ctx.config['batch_size']

    control_emb, case_emb = [], []
    with torch.no_grad():
        for _ in tqdm(range(test_steps), desc='perturbation_detection: Extracting embeddings'):
            batch = test_loader.next_batch()
            cont_x, cont_tot = batch.control, batch.control_total
            case_x, case_tot = batch.case, batch.case_total
            ctrl_z = ctx.biojepa.student(cont_x, cont_tot, mask_idx=None).mean(dim=1).cpu().numpy()
            case_z = ctx.biojepa.student(case_x, case_tot, mask_idx=None).mean(dim=1).cpu().numpy()
            control_emb.append(ctrl_z)
            case_emb.append(case_z)

    control_emb = np.concatenate(control_emb, axis=0)
    case_emb = np.concatenate(case_emb, axis=0)

    X = np.concatenate([control_emb, case_emb], axis=0)
    y = np.concatenate([np.zeros(len(control_emb)), np.ones(len(case_emb))]).astype(int)

    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)

    print('Training perturbation detector...')
    classifier, val_preds, val_acc = train_linear_classifier(X[train_idx], y[train_idx], X[val_idx], y[val_idx], num_classes=2, device=ctx.device, epochs=100)

    classifier.eval()
    with torch.no_grad():
        X_val_t = torch.from_numpy(X[val_idx]).float().to(ctx.device)
        logits = classifier(X_val_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    auroc = roc_auc_score(y[val_idx], probs)

    print(f'perturbation_detection: AUROC={auroc:.4f}, Accuracy={val_acc:.4f}')

    return {
        'config': {'n_control': len(control_emb), 'n_perturbed': len(case_emb), 'embedding_dim': int(control_emb.shape[1])},
        'metrics': {'auroc': float(auroc), 'accuracy': float(val_acc), 'chance': 0.5}
    }


def _embedding_consistency(ctx):
    '''Do replicates of the same perturbation cluster together?'''
    test_loader = TrainingLoader(batch_size=ctx.config['batch_size'], split='test', data_dir=ctx.paths['train_dir'], device=ctx.device)
    test_steps = ctx.config['test_total_examples'] // ctx.config['batch_size']

    all_emb, all_pert = [], []
    with torch.no_grad():
        for _ in tqdm(range(test_steps), desc='embedding_consistency: Extracting embeddings'):
            batch = test_loader.next_batch()
            case_x, case_tot = batch.case, batch.case_total
            case_z = ctx.biojepa.student(case_x, case_tot, mask_idx=None).mean(dim=1).cpu().numpy()
            all_emb.append(case_z)
            all_pert.append(batch.seq_idx[:, 0].cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0)
    pert_ids = np.concatenate(all_pert, axis=0).flatten()

    pert_to_emb = defaultdict(list)
    for i, pid in enumerate(pert_ids):
        pert_to_emb[pid].append(embeddings[i])

    valid_perts = {pid: np.array(embs) for pid, embs in pert_to_emb.items() if len(embs) >= 3}

    if len(valid_perts) < 10:
        return {'error': f'Not enough perturbations with >= 3 replicates (found {len(valid_perts)})'}

    intra_dists = []
    for pid, embs in valid_perts.items():
        n = len(embs)
        for i in range(n):
            for j in range(i + 1, n):
                intra_dists.append(np.linalg.norm(embs[i] - embs[j]))

    pert_list = list(valid_perts.keys())
    inter_dists = []
    n_samples = min(5000, len(pert_list) * (len(pert_list) - 1) // 2)
    rng = np.random.RandomState(42)
    for _ in range(n_samples):
        p1, p2 = rng.choice(pert_list, 2, replace=False)
        e1 = valid_perts[p1][rng.randint(len(valid_perts[p1]))]
        e2 = valid_perts[p2][rng.randint(len(valid_perts[p2]))]
        inter_dists.append(np.linalg.norm(e1 - e2))

    intra_mean, inter_mean = np.mean(intra_dists), np.mean(inter_dists)
    ratio = inter_mean / intra_mean if intra_mean > 0 else float('inf')

    print(f'embedding_consistency: Intra={intra_mean:.4f}, Inter={inter_mean:.4f}, Ratio={ratio:.2f}x')

    return {
        'config': {'n_perturbations': len(valid_perts), 'n_intra_pairs': len(intra_dists), 'n_inter_pairs': len(inter_dists)},
        'metrics': {'mean_intra_distance': float(intra_mean), 'mean_inter_distance': float(inter_mean), 'inter_intra_ratio': float(ratio), 'std_intra_distance': float(np.std(intra_dists)), 'std_inter_distance': float(np.std(inter_dists))}
    }


def _latent_space_health(ctx):
    '''Diagnostic metrics for embedding quality.'''
    test_loader = TrainingLoader(batch_size=ctx.config['batch_size'], split='test', data_dir=ctx.paths['train_dir'], device=ctx.device)
    test_steps = ctx.config['test_total_examples'] // ctx.config['batch_size']

    all_emb = []
    with torch.no_grad():
        for _ in tqdm(range(test_steps), desc='latent_space_health: Extracting embeddings'):
            batch = test_loader.next_batch()
            cont_x, cont_tot = batch.control, batch.control_total
            emb = ctx.biojepa.student(cont_x, cont_tot, mask_idx=None).mean(dim=1).cpu().numpy()
            all_emb.append(emb)

    embeddings = np.concatenate(all_emb, axis=0)
    N, D = embeddings.shape

    dim_variances = np.var(embeddings, axis=0)
    mean_variance, min_variance, max_variance = np.mean(dim_variances), np.min(dim_variances), np.max(dim_variances)
    n_dead_dims = int(np.sum(dim_variances < 1e-6))

    pca = PCA()
    pca.fit(embeddings)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    effective_dim_90 = int(np.searchsorted(cumsum, 0.90) + 1)
    effective_dim_95 = int(np.searchsorted(cumsum, 0.95) + 1)

    eigenvalues = pca.explained_variance_
    isotropy_ratio = float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else 0.0

    n_sample = min(1000, N)
    sample_idx = np.random.RandomState(42).choice(N, n_sample, replace=False)
    sample_emb = embeddings[sample_idx]
    sample_emb_norm = sample_emb / (np.linalg.norm(sample_emb, axis=1, keepdims=True) + 1e-8)
    cos_sim_matrix = sample_emb_norm @ sample_emb_norm.T
    upper_tri = cos_sim_matrix[np.triu_indices(n_sample, k=1)]
    mean_cos_sim = float(np.mean(upper_tri))

    print(f'latent_space_health: Eff_dim_90={effective_dim_90}/{D}, Mean_var={mean_variance:.4f}, Isotropy={isotropy_ratio:.6f}')

    return {
        'config': {'samples': N, 'embedding_dim': D},
        'effective_dimensionality': {'90_percent': effective_dim_90, '95_percent': effective_dim_95, 'ratio_to_total_90': float(effective_dim_90 / D)},
        'variance': {'mean': float(mean_variance), 'min': float(min_variance), 'max': float(max_variance), 'n_dead_dims': n_dead_dims},
        'isotropy': {'min_max_eigenvalue_ratio': isotropy_ratio, 'mean_pairwise_cosine_sim': mean_cos_sim}
    }


# =============================================================================
# FULL MODEL EVALS
# =============================================================================

def _expression_prediction(ctx):
    '''Can we predict gene expression after perturbation?'''
    inf = ctx.test_inference
    pert_keys = inf['pert_keys']
    mean_pred_deltas, mean_real_deltas = inf['mean_pred_deltas'], inf['mean_real_deltas']
    mean_pred_abs, mean_real_abs = inf['mean_pred_abs'], inf['mean_real_abs']
    sample_mses, sample_correlations = inf['sample_mses'], inf['sample_correlations']
    n_genes = ctx.config['num_genes']

    TOP_K = 50
    per_pert_r2_all, per_pert_r2_top50, per_pert_mse = [], [], []

    for key in pert_keys:
        pred_abs, real_abs = mean_pred_abs[key], mean_real_abs[key]
        pred_delta, real_delta = mean_pred_deltas[key], mean_real_deltas[key]

        if np.std(real_abs) > 1e-9:
            per_pert_r2_all.append(r2_score(real_abs, pred_abs))
        top_k_idx = np.argsort(np.abs(real_delta))[-TOP_K:]
        per_pert_r2_top50.append(r2_score(real_abs[top_k_idx], pred_abs[top_k_idx]))
        per_pert_mse.append(np.mean((pred_delta - real_delta)**2))

    per_pert_r2_all, per_pert_r2_top50 = np.array(per_pert_r2_all), np.array(per_pert_r2_top50)
    per_pert_mse = np.array(per_pert_mse)

    pred_severity = np.array([np.linalg.norm(mean_pred_deltas[k]) for k in pert_keys])
    real_severity = np.array([np.linalg.norm(mean_real_deltas[k]) for k in pert_keys])
    severity_pearson, _ = pearsonr(pred_severity, real_severity)
    severity_spearman, _ = spearmanr(pred_severity, real_severity)

    all_pred = np.concatenate([mean_pred_deltas[k] for k in pert_keys])
    all_real = np.concatenate([mean_real_deltas[k] for k in pert_keys])
    all_errors, all_magnitudes = all_pred - all_real, np.abs(all_real)

    magnitude_bins = [0, 0.25, 0.5, 1.0, 1.5, 2.0, np.inf]
    bin_labels = ['0-0.25', '0.25-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0+']
    error_by_magnitude = {}
    for i in range(len(magnitude_bins) - 1):
        mask = (all_magnitudes >= magnitude_bins[i]) & (all_magnitudes < magnitude_bins[i + 1])
        if mask.sum() > 0:
            error_by_magnitude[bin_labels[i]] = {'mae': float(np.mean(np.abs(all_errors[mask]))), 'count': int(mask.sum())}

    print(f'expression_prediction: MSE={np.mean(sample_mses):.4f}, R2_all={per_pert_r2_all.mean():.4f}')

    return {
        'config': {'test_perturbations': len(pert_keys), 'genes': n_genes, 'test_samples': len(sample_mses)},
        'sample_level': {'mse': float(np.mean(sample_mses)), 'pearson_r_top20': float(np.mean(sample_correlations))},
        'perturbation_level': {
            'r2_all_genes': {'mean': float(per_pert_r2_all.mean()), 'median': float(np.median(per_pert_r2_all))},
            'r2_top50_degs': {'mean': float(per_pert_r2_top50.mean()), 'median': float(np.median(per_pert_r2_top50))},
            'mse': {'mean': float(per_pert_mse.mean()), 'median': float(np.median(per_pert_mse))}
        },
        'severity': {'pearson_r': float(severity_pearson), 'spearman_r': float(severity_spearman)},
        'error_by_magnitude': error_by_magnitude
    }


def _gene_level_analysis(ctx, direction_threshold=0.25):
    '''Direction of effect + top DEG recovery analysis.'''
    inf = ctx.test_inference
    pert_keys = inf['pert_keys']
    mean_pred_deltas, mean_real_deltas = inf['mean_pred_deltas'], inf['mean_real_deltas']
    n_genes = ctx.config['num_genes']

    def classify_direction(delta, threshold=direction_threshold):
        direction = np.zeros_like(delta, dtype=np.int8)
        direction[delta >= threshold] = 1
        direction[delta <= -threshold] = -1
        return direction

    all_pred_dir = np.concatenate([classify_direction(mean_pred_deltas[k]) for k in pert_keys])
    all_real_dir = np.concatenate([classify_direction(mean_real_deltas[k]) for k in pert_keys])
    overall_accuracy = accuracy_score(all_real_dir, all_pred_dir)
    f1_up = f1_score(all_real_dir, all_pred_dir, labels=[1], average='macro', zero_division=0)
    f1_down = f1_score(all_real_dir, all_pred_dir, labels=[-1], average='macro', zero_division=0)
    f1_unchanged = f1_score(all_real_dir, all_pred_dir, labels=[0], average='macro', zero_division=0)

    TOP_K_DIR = 50
    top_deg_pred, top_deg_real = [], []
    for key in pert_keys:
        top_k_idx = np.argsort(np.abs(mean_real_deltas[key]))[-TOP_K_DIR:]
        top_deg_pred.append(classify_direction(mean_pred_deltas[key][top_k_idx]))
        top_deg_real.append(classify_direction(mean_real_deltas[key][top_k_idx]))
    top_deg_accuracy = accuracy_score(np.concatenate(top_deg_real), np.concatenate(top_deg_pred))

    all_magnitudes, all_correct = [], []
    for key in pert_keys:
        real_delta, pred_delta = mean_real_deltas[key], mean_pred_deltas[key]
        all_magnitudes.extend(np.abs(real_delta))
        all_correct.extend(classify_direction(pred_delta) == classify_direction(real_delta))
    all_magnitudes, all_correct = np.array(all_magnitudes), np.array(all_correct)

    magnitude_bins = [0, 0.25, 0.5, 1.0, 1.5, 2.0, np.inf]
    bin_labels = ['0-0.25', '0.25-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0+']
    accuracy_by_magnitude = {}
    for i in range(len(magnitude_bins) - 1):
        mask = (all_magnitudes >= magnitude_bins[i]) & (all_magnitudes < magnitude_bins[i + 1])
        if mask.sum() > 0:
            accuracy_by_magnitude[bin_labels[i]] = {'accuracy': float(all_correct[mask].mean()), 'count': int(mask.sum())}

    def precision_at_k(pred_rank, true_rank, k):
        return len(set(pred_rank[:k]) & set(true_rank[:k])) / k

    def ndcg_at_k(pred_rank, true_rank, k):
        true_set = set(true_rank[:k])
        rels = [1 if g in true_set else 0 for g in pred_rank[:k]]
        dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rels))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(true_set))))
        return dcg / idcg if idcg > 0 else 0.0

    K_VALUES = [10, 20, 50, 100]
    deg_results = {k: {'precision': [], 'ndcg': [], 'overlap': []} for k in K_VALUES}
    for key in pert_keys:
        pred_rank = np.argsort(np.abs(mean_pred_deltas[key]))[::-1]
        true_rank = np.argsort(np.abs(mean_real_deltas[key]))[::-1]
        for k in K_VALUES:
            deg_results[k]['precision'].append(precision_at_k(pred_rank, true_rank, k))
            deg_results[k]['ndcg'].append(ndcg_at_k(pred_rank, true_rank, k))
            deg_results[k]['overlap'].append(len(set(pred_rank[:k]) & set(true_rank[:k])))

    print(f'gene_level_analysis: Dir_acc={overall_accuracy:.4f}, Top50_acc={top_deg_accuracy:.4f}')

    return {
        'config': {'test_perturbations': len(pert_keys), 'genes': n_genes, 'direction_threshold': direction_threshold},
        'direction_of_effect': {'all_genes_accuracy': float(overall_accuracy), 'top50_degs_accuracy': float(top_deg_accuracy), 'f1_up': float(f1_up), 'f1_down': float(f1_down), 'f1_unchanged': float(f1_unchanged), 'accuracy_by_magnitude': accuracy_by_magnitude},
        'top_deg_recovery': {str(k): {'precision': float(np.mean(deg_results[k]['precision'])), 'ndcg': float(np.mean(deg_results[k]['ndcg'])), 'overlap': float(np.mean(deg_results[k]['overlap'])), 'vs_random': float(np.mean(deg_results[k]['overlap']) / (k * k / n_genes))} for k in K_VALUES}
    }


def _perturbation_retrieval(ctx, n_eval=100):
    '''Given desired outcome, can we find the right perturbation?'''
    inf = ctx.test_inference
    pert_keys = inf['pert_keys']
    mean_real_deltas = inf['mean_real_deltas']
    mean_control_states = inf['mean_control_states']
    n_genes = ctx.config['num_genes']

    retrieval_banks = {}
    if ctx.seq_banks and 'dna' in ctx.seq_banks:
        dna = ctx.seq_banks['dna']
        if dna.shape[-1] < MAX_SEQ_DIM:
            dna = F.pad(dna, (0, MAX_SEQ_DIM - dna.shape[-1]))
        retrieval_banks['dna'] = {'bank': dna, 'mod_id': 0, 'mode': 0, 'use_seq': True}
    if ctx.seq_banks and 'chemical' in ctx.seq_banks:
        chem = ctx.seq_banks['chemical']
        if chem.shape[-1] < MAX_SEQ_DIM:
            chem = F.pad(chem, (0, MAX_SEQ_DIM - chem.shape[-1]))
        retrieval_banks['chemical'] = {'bank': chem, 'mod_id': 2, 'mode': 4, 'use_seq': True}
    if ctx.target_bank is not None:
        retrieval_banks['target_only'] = {'bank': ctx.target_bank, 'mod_id': 0, 'mode': 0, 'use_seq': False}

    if not retrieval_banks:
        return {'error': 'No perturbation banks available'}

    def predict_all_deltas(control_x_np, bank_info, batch_size=64):
        bank, mod_id, mode_id, use_seq = bank_info['bank'], bank_info['mod_id'], bank_info['mode'], bank_info['use_seq']
        n_perts = bank.shape[0]
        control_x = torch.from_numpy(control_x_np).float().to(ctx.device)
        control_tot = control_x.sum()
        all_pred = []
        for start in range(0, n_perts, batch_size):
            end = min(start + batch_size, n_perts)
            B = end - start
            batch_idx = torch.arange(start, end, device=ctx.device)
            with torch.no_grad():
                control_batch = control_x.unsqueeze(0).expand(B, -1)
                control_tot_batch = control_tot.unsqueeze(0).expand(B)
                z_ctx = ctx.biojepa.student(control_batch, control_tot_batch, mask_idx=None)
                if use_seq:
                    seq_emb = bank[batch_idx].unsqueeze(1)
                    target_emb = torch.zeros(B, 1, 320, device=ctx.device)
                    has_seq = torch.ones(B, 1, dtype=torch.bool, device=ctx.device)
                    has_target = torch.zeros(B, 1, dtype=torch.bool, device=ctx.device)
                else:
                    seq_emb = torch.zeros(B, 1, MAX_SEQ_DIM, device=ctx.device)
                    target_emb = bank[batch_idx].unsqueeze(1)
                    has_seq = torch.zeros(B, 1, dtype=torch.bool, device=ctx.device)
                    has_target = torch.ones(B, 1, dtype=torch.bool, device=ctx.device)
                modality_ids = torch.full((B, 1), mod_id, dtype=torch.long, device=ctx.device)
                mode_ids = torch.full((B, 1), mode_id, dtype=torch.long, device=ctx.device)
                pert_mask = torch.ones(B, 1, dtype=torch.bool, device=ctx.device)
                action = ctx.biojepa.composer(seq_emb, target_emb, modality_ids, mode_ids, has_seq, has_target, pert_mask)
                targets = torch.arange(n_genes, device=ctx.device).unsqueeze(0).expand(B, -1)
                z_pred, _ = ctx.biojepa.predictor(z_ctx, action, targets)
                all_pred.append((ctx.decoder(z_pred) - ctx.decoder(z_ctx)).cpu().numpy())
        return np.concatenate(all_pred, axis=0)

    def cos_sim(a, b):
        a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
        b_n = b / (np.linalg.norm(b) + 1e-8)
        return np.dot(a_n, b_n)

    results_by_type = {}
    K_VALUES = [1, 5, 10, 20, 50]

    for bank_name, bank_info in retrieval_banks.items():
        n_bank = bank_info['bank'].shape[0]
        if bank_name == 'target_only':
            type_keys = [k for k in pert_keys if k[0] < 0 and k[1] >= 0]
            idx_pos = 1
        else:
            mod_id = bank_info['mod_id']
            type_keys = [k for k in pert_keys if k[0] >= 0 and k[2] == mod_id]
            idx_pos = 0

        eval_keys = type_keys[:min(len(type_keys), n_eval)]
        if not eval_keys:
            results_by_type[bank_name] = {'n_test': 0, 'n_evaluated': 0}
            continue

        ranks = []
        for key in tqdm(eval_keys, desc=f'perturbation_retrieval ({bank_name})'):
            lookup_idx = key[idx_pos]
            if lookup_idx < 0 or lookup_idx >= n_bank:
                continue
            preds = predict_all_deltas(mean_control_states[key], bank_info)
            sims = cos_sim(preds, mean_real_deltas[key])
            rank = int(np.where(np.argsort(sims)[::-1] == lookup_idx)[0][0]) + 1
            ranks.append(rank)

        if ranks:
            ranks = np.array(ranks)
            results_by_type[bank_name] = {
                'mrr': float(np.mean(1.0/ranks)),
                'median_rank': float(np.median(ranks)),
                'n_evaluated': len(ranks),
                'n_bank': n_bank,
                'recall_at_k': {str(k): float(np.mean(ranks <= k)) for k in K_VALUES if k <= n_bank}
            }
            print(f'perturbation_retrieval ({bank_name}): MRR={results_by_type[bank_name]["mrr"]:.4f}')
        else:
            results_by_type[bank_name] = {'n_test': 0, 'n_evaluated': 0}

    return {'by_type': results_by_type}


def _uncertainty_calibration(ctx, n_bins=10):
    '''Are confidence estimates meaningful?'''
    inf = ctx.test_inference
    pred_deltas = inf['sample_pred_deltas']
    real_deltas = inf['sample_real_deltas']
    sample_logvars = inf['sample_logvars']
    pert_ids = inf['sample_pert_ids']
    target_ids = inf['sample_target_ids']
    pert_mods = inf['sample_pert_mods']

    sample_mse = np.mean((pred_deltas - real_deltas)**2, axis=1)
    sample_unc = sample_logvars.mean(axis=1)

    pearson_r, _ = pearsonr(sample_unc, sample_mse)
    spearman_r, _ = spearmanr(sample_unc, sample_mse)

    bin_edges = np.percentile(sample_unc, np.linspace(0, 100, n_bins + 1))
    bin_mean_error = []
    for i in range(n_bins):
        mask = (sample_unc >= bin_edges[i]) & (sample_unc <= bin_edges[i + 1] if i == n_bins - 1 else sample_unc < bin_edges[i + 1])
        bin_mean_error.append(sample_mse[mask].mean() if mask.sum() > 0 else 0)

    monotonicity = sum(1 for i in range(len(bin_mean_error) - 1) if bin_mean_error[i + 1] > bin_mean_error[i]) / (n_bins - 1)

    unc_norm = (sample_unc - sample_unc.min()) / (sample_unc.max() - sample_unc.min() + 1e-8)
    err_norm = (sample_mse - sample_mse.min()) / (sample_mse.max() - sample_mse.min() + 1e-8)
    ece = 0.0
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins if i < n_bins - 1 else 1.0
        mask = (unc_norm >= low) & (unc_norm < high if i < n_bins - 1 else unc_norm <= high)
        if mask.sum() == 0:
            continue
        bin_weight = mask.sum() / len(sample_mse)
        bin_unc_mean = unc_norm[mask].mean()
        bin_err_mean = err_norm[mask].mean()
        ece += bin_weight * abs(bin_unc_mean - bin_err_mean)

    pert_unc, pert_err = defaultdict(list), defaultdict(list)
    for i in range(len(pert_ids)):
        key = (int(pert_ids[i]), int(target_ids[i]), int(pert_mods[i]))
        pert_unc[key].append(sample_unc[i])
        pert_err[key].append(sample_mse[i])
    pert_unc_arr = np.array([np.mean(pert_unc[p]) for p in pert_unc])
    pert_err_arr = np.array([np.mean(pert_err[p]) for p in pert_err])
    pert_pearson, _ = pearsonr(pert_unc_arr, pert_err_arr)
    pert_spearman, _ = spearmanr(pert_unc_arr, pert_err_arr)

    print(f'uncertainty_calibration: ECE={ece:.4f}, Monotonicity={monotonicity:.2%}')

    return {
        'config': {'samples': len(sample_mse), 'perturbations': len(pert_unc)},
        'sample_level': {'uncertainty_error_pearson': float(pearson_r), 'uncertainty_error_spearman': float(spearman_r), 'expected_calibration_error': float(ece), 'monotonicity_score': float(monotonicity)},
        'perturbation_level': {'uncertainty_error_pearson': float(pert_pearson), 'uncertainty_error_spearman': float(pert_spearman)},
        'bin_analysis': {'n_bins': n_bins, 'bin_mean_errors': [float(e) for e in bin_mean_error]}
    }


def _action_vector_pathways(ctx):
    '''Do perturbations targeting same pathway produce similar action vectors?'''
    pathway_libs = ctx.pathway_annotations
    inf = ctx.alignment_inference

    gene_to_pathway = {}
    for pathway, genes in pathway_libs['KEGG_2021_Human'].items():
        if 15 <= len(genes) <= 300:
            for gene in genes:
                if gene.upper() not in gene_to_pathway:
                    gene_to_pathway[gene.upper()] = pathway

    results = {}

    if 'dna_actions' in inf:
        dna_actions = inf['dna_actions'].cpu().numpy()
        id_to_gene = ctx.id_to_gene
        pert_labels = {pid: gene_to_pathway[gene] for pid, gene in id_to_gene.items() if gene in gene_to_pathway and pid < dna_actions.shape[0]}
        if len(pert_labels) >= 10:
            action_idx = list(pert_labels.keys())
            metrics = compute_pathway_clustering_metrics(dna_actions[action_idx], [pert_labels[i] for i in action_idx], min_samples_per_class=5)
            results['dna'] = metrics
            print(f'action_vector_pathways DNA: sil={metrics.get("silhouette_score", "N/A")}')

    if 'chem_actions' in inf:
        chem_actions = inf['chem_actions'].cpu().numpy()
        pairs = ctx.alignment_pairs
        chem_mask = pairs['modality'] == 2
        if chem_mask.sum() > 0:
            chem_seq_idx = pairs['seq_idx'][chem_mask]
            chem_target_idx = pairs['target_idx'][chem_mask]
            gene_to_target_path = ctx.paths['pert_dir'] / 'target_banks' / 'gene_to_target_idx.json'
            if gene_to_target_path.exists():
                with open(gene_to_target_path) as f:
                    gene_to_target = json.load(f)
                target_to_gene = {tidx: gene.upper() for gene, tidx in gene_to_target.items() if not gene.startswith('ENSG')}
                chem_pert_labels = {}
                for s, t in zip(chem_seq_idx, chem_target_idx):
                    if s < chem_actions.shape[0] and t in target_to_gene:
                        gene = target_to_gene[t]
                        if gene in gene_to_pathway:
                            chem_pert_labels[int(s)] = gene_to_pathway[gene]
                if len(chem_pert_labels) >= 10:
                    action_idx = list(chem_pert_labels.keys())
                    metrics = compute_pathway_clustering_metrics(chem_actions[action_idx], [chem_pert_labels[i] for i in action_idx], min_samples_per_class=5)
                    results['chemical'] = metrics
                    print(f'action_vector_pathways chemical: sil={metrics.get("silhouette_score", "N/A")}')

    if not results:
        return {'error': 'No action vectors available for pathway analysis'}

    return {'config': {'n_pathways_kegg': len(gene_to_pathway)}, 'by_modality': results}


def _moa_matching(ctx):
    '''Do same-pathway perturbations produce similar predicted effects?'''
    inf = ctx.test_inference
    pathway_libs = ctx.pathway_annotations
    id_to_gene = ctx.id_to_gene

    gene_to_pathway = {}
    for pathway, genes in pathway_libs['KEGG_2021_Human'].items():
        if 15 <= len(genes) <= 200:
            for gene in genes:
                if gene.upper() not in gene_to_pathway:
                    gene_to_pathway[gene.upper()] = pathway

    test_keys = set(inf['pert_keys'])
    seq_idx_to_keys = defaultdict(list)
    for key in test_keys:
        seq_idx_to_keys[key[0]].append(key)

    valid_keys = []
    key_to_pathway = {}
    for seq_idx, gene in id_to_gene.items():
        if gene in gene_to_pathway and seq_idx in seq_idx_to_keys:
            for key in seq_idx_to_keys[seq_idx]:
                valid_keys.append(key)
                key_to_pathway[key] = gene_to_pathway[gene]

    pathway_to_keys = defaultdict(list)
    for key in valid_keys:
        pathway_to_keys[key_to_pathway[key]].append(key)
    valid_pathways = {p: keys for p, keys in pathway_to_keys.items() if len(keys) >= 3}

    if len(valid_pathways) < 2:
        return {'error': 'Not enough valid pathways'}

    all_keys = [k for keys in valid_pathways.values() for k in keys]
    key_to_idx = {k: i for i, k in enumerate(all_keys)}
    delta_matrix = np.array([inf['mean_pred_deltas'][k] for k in all_keys])
    sim_matrix = cosine_similarity(delta_matrix)

    within_sims, between_sims = [], []
    for pathway, keys in valid_pathways.items():
        idx = [key_to_idx[k] for k in keys]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                within_sims.append(sim_matrix[idx[i], idx[j]])

    pathways = list(valid_pathways.keys())
    for i, p1 in enumerate(pathways):
        for j in range(i + 1, len(pathways)):
            p2 = pathways[j]
            for k1 in valid_pathways[p1]:
                for k2 in valid_pathways[p2]:
                    between_sims.append(sim_matrix[key_to_idx[k1], key_to_idx[k2]])

    within_sims, between_sims = np.array(within_sims), np.array(between_sims)
    mean_within, mean_between = np.mean(within_sims), np.mean(between_sims)
    ratio = mean_within / mean_between if mean_between != 0 else float('inf')
    _, p_val = mannwhitneyu(within_sims, between_sims, alternative='greater')

    print(f'moa_matching: Within={mean_within:.4f}, Between={mean_between:.4f}, Ratio={ratio:.3f}x')

    return {
        'config': {'n_pathways': len(valid_pathways), 'n_perturbations': len(all_keys)},
        'similarity': {'mean_within_pathway': float(mean_within), 'mean_between_pathway': float(mean_between), 'similarity_ratio': float(ratio), 'mann_whitney_p': float(p_val), 'n_within_pairs': len(within_sims), 'n_between_pairs': len(between_sims)}
    }


# =============================================================================
# ALIGNMENT EVALS (v0.6 dual-path architecture)
# =============================================================================

def _seq_to_target_retrieval(ctx):
    '''Per-modality retrieval: can we retrieve correct protein target from sequence query?'''
    pairs = ctx.alignment_pairs
    inf = ctx.alignment_inference
    if 'target_actions_norm' not in inf:
        return {'error': 'No target embeddings available'}

    target_norm = inf['target_actions_norm']
    n_targets = target_norm.shape[0]
    K_VALUES = [1, 5, 10, 20, 50]
    results = {}

    for modality_name, modality_id in [('dna', 0), ('chemical', 2)]:
        action_key = f'{modality_name}_actions_norm'
        if action_key not in inf:
            continue

        seq_norm = inf[action_key]
        mod_mask = pairs['modality'] == modality_id
        if mod_mask.sum() == 0:
            continue

        mod_seq_idx = pairs['seq_idx'][mod_mask]
        mod_target_idx = pairs['target_idx'][mod_mask]

        unique_seqs = np.unique(mod_seq_idx)
        seq_to_targets = {s: [] for s in unique_seqs}
        for s, t in zip(mod_seq_idx, mod_target_idx):
            seq_to_targets[s].append(t)

        sim_matrix = torch.mm(seq_norm, target_norm.T).cpu().numpy()
        ranks, reciprocal_ranks = [], []
        recall_at_k = {k: [] for k in K_VALUES}

        for seq_id in unique_seqs:
            if seq_id >= sim_matrix.shape[0]:
                continue
            correct_targets = set(seq_to_targets[seq_id])
            sims = sim_matrix[seq_id]
            sorted_indices = np.argsort(sims)[::-1]
            for rank, idx in enumerate(sorted_indices, 1):
                if idx in correct_targets:
                    ranks.append(rank)
                    reciprocal_ranks.append(1.0 / rank)
                    for k in K_VALUES:
                        recall_at_k[k].append(1 if rank <= k else 0)
                    break

        if not ranks:
            continue

        results[modality_name] = {
            'mrr': float(np.mean(reciprocal_ranks)),
            'median_rank': float(np.median(ranks)),
            'mean_rank': float(np.mean(ranks)),
            'n_queries': len(unique_seqs),
            'n_targets': n_targets,
            'recall_at_k': {str(k): float(np.mean(recall_at_k[k])) for k in K_VALUES}
        }

    print(f'seq_to_target_retrieval: ' + ', '.join(f'{m}_mrr={r["mrr"]:.4f}' for m, r in results.items()))
    return {'config': {'n_targets': n_targets}, 'by_modality': results}


def _cross_modality_target_consistency(ctx):
    '''Different sequences targeting same protein should produce similar actions.'''
    pairs = ctx.alignment_pairs
    inf = ctx.alignment_inference
    if 'dna_actions_norm' not in inf:
        return {'error': 'No DNA embeddings available'}

    dna_norm = inf['dna_actions_norm']
    target_to_dna = defaultdict(list)
    dna_mask = pairs['modality'] == 0
    for seq_idx, target_idx in zip(pairs['seq_idx'][dna_mask], pairs['target_idx'][dna_mask]):
        if seq_idx < dna_norm.shape[0]:
            target_to_dna[target_idx].append(seq_idx)

    valid_targets = {t: seqs for t, seqs in target_to_dna.items() if len(seqs) >= 2}
    if len(valid_targets) < 5:
        return {'error': f'Not enough targets with multiple sequences (found {len(valid_targets)})'}

    within_target_sims = []
    for target, seq_indices in valid_targets.items():
        seq_actions = dna_norm[seq_indices].cpu().numpy()
        sim_matrix = cosine_similarity(seq_actions)
        n = len(seq_indices)
        for i in range(n):
            for j in range(i + 1, n):
                within_target_sims.append(sim_matrix[i, j])

    rng = np.random.RandomState(42)
    between_target_sims = []
    target_list = list(valid_targets.keys())
    n_samples = min(5000, len(target_list) * (len(target_list) - 1) // 2)
    for _ in range(n_samples):
        t1, t2 = rng.choice(target_list, 2, replace=False)
        s1 = rng.choice(valid_targets[t1])
        s2 = rng.choice(valid_targets[t2])
        if s1 < dna_norm.shape[0] and s2 < dna_norm.shape[0]:
            sim = torch.dot(dna_norm[s1], dna_norm[s2]).item()
            between_target_sims.append(sim)

    within_mean = float(np.mean(within_target_sims))
    between_mean = float(np.mean(between_target_sims))
    consistency_ratio = within_mean / between_mean if between_mean != 0 else float('inf')

    print(f'cross_modality_target_consistency: Within={within_mean:.4f}, Between={between_mean:.4f}, Ratio={consistency_ratio:.2f}x')

    return {
        'config': {'n_valid_targets': len(valid_targets), 'n_within_pairs': len(within_target_sims), 'n_between_pairs': len(between_target_sims)},
        'metrics': {'within_target_sim': within_mean, 'between_target_sim': between_mean, 'consistency_ratio': consistency_ratio}
    }


def _seq_target_gap_analysis(ctx):
    '''Per-modality gap between sequence and target representation spaces.'''
    inf = ctx.alignment_inference
    if 'target_actions' not in inf:
        return {'error': 'No target embeddings available'}

    target_actions = inf['target_actions'].cpu().numpy()
    target_centroid = np.mean(target_actions, axis=0)
    target_var = float(np.mean([np.linalg.norm(a - target_centroid)**2 for a in target_actions]))

    results = {'target_variance': target_var, 'n_targets': len(target_actions)}
    rng = np.random.RandomState(42)
    n_sample = min(500, len(target_actions))
    target_sample = target_actions[rng.choice(len(target_actions), n_sample, replace=False)]

    for modality_name in ['dna', 'chemical']:
        action_key = f'{modality_name}_actions'
        if action_key not in inf:
            continue

        seq_actions = inf[action_key].cpu().numpy()
        seq_centroid = np.mean(seq_actions, axis=0)
        seq_var = float(np.mean([np.linalg.norm(a - seq_centroid)**2 for a in seq_actions]))
        centroid_dist = float(np.linalg.norm(seq_centroid - target_centroid))

        n_seq_sample = min(500, len(seq_actions))
        seq_sample = seq_actions[rng.choice(len(seq_actions), n_seq_sample, replace=False)]
        within_seq = pdist(seq_sample)
        between = np.linalg.norm(seq_sample[:, None] - target_sample[None, :], axis=-1).flatten()
        gap_ratio = float(np.mean(between) / np.mean(within_seq)) if np.mean(within_seq) > 0 else float('inf')

        results[modality_name] = {
            'seq_variance': seq_var,
            'centroid_distance': centroid_dist,
            'mean_within_seq': float(np.mean(within_seq)),
            'mean_seq_to_target': float(np.mean(between)),
            'gap_ratio': gap_ratio,
            'n_sequences': len(seq_actions)
        }

    print(f'seq_target_gap_analysis: ' + ', '.join(f'{m}_gap={r["gap_ratio"]:.2f}' for m, r in results.items() if isinstance(r, dict)))
    return results


def _paired_alignment_quality(ctx):
    '''Direct cosine similarity for known seq-target pairs, by modality.'''
    pairs = ctx.alignment_pairs
    inf = ctx.alignment_inference
    if 'target_actions_norm' not in inf:
        return {'error': 'No target embeddings available'}

    target_norm = inf['target_actions_norm']
    results = {}

    for modality_name, modality_id in [('dna', 0), ('chemical', 2)]:
        action_key = f'{modality_name}_actions_norm'
        if action_key not in inf:
            continue

        seq_norm = inf[action_key]
        mod_mask = pairs['modality'] == modality_id
        if mod_mask.sum() == 0:
            continue

        seq_idx = pairs['seq_idx'][mod_mask]
        target_idx = pairs['target_idx'][mod_mask]

        cos_sims = []
        for s, t in zip(seq_idx, target_idx):
            if s < seq_norm.shape[0] and t < target_norm.shape[0]:
                sim = torch.dot(seq_norm[s], target_norm[t]).item()
                cos_sims.append(sim)

        if not cos_sims:
            continue

        cos_sims = np.array(cos_sims)
        results[modality_name] = {
            'mean_cosine_sim': float(np.mean(cos_sims)),
            'std_cosine_sim': float(np.std(cos_sims)),
            'n_pairs': len(cos_sims),
            'percentiles': {str(p): float(np.percentile(cos_sims, p)) for p in [5, 25, 50, 75, 95]}
        }

    print(f'paired_alignment_quality: ' + ', '.join(f'{m}_sim={r["mean_cosine_sim"]:.4f}' for m, r in results.items()))
    return {'by_modality': results}


def _mode_sensitivity(ctx):
    '''Does FiLM conditioning on mode differentiate perturbation effects?'''
    inf = ctx.alignment_inference
    seq_bank = ctx.seq_banks.get('dna') if ctx.seq_banks else None
    if seq_bank is None:
        return {'error': 'No DNA sequence bank available'}

    n_sample = min(200, seq_bank.shape[0])
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(seq_bank.shape[0], n_sample, replace=False)

    GENETIC_MODES = {'crispri': 0, 'crispra': 1, 'overexpression': 2, 'knockout': 3}
    DRUG_MODES = {'inhibitor': 4, 'agonist': 5, 'degrader': 6}
    ALL_MODES = {**GENETIC_MODES, **DRUG_MODES}

    mode_actions = {}
    with torch.no_grad():
        seq_emb = torch.zeros(n_sample, 1, MAX_SEQ_DIM, device=ctx.device)
        emb = seq_bank[sample_idx]
        if emb.shape[-1] < MAX_SEQ_DIM:
            emb = F.pad(emb, (0, MAX_SEQ_DIM - emb.shape[-1]))
        seq_emb[:, 0, :] = emb
        modality_ids = torch.zeros(n_sample, 1, dtype=torch.long, device=ctx.device)
        pert_mask = torch.ones(n_sample, 1, dtype=torch.bool, device=ctx.device)

        for mode_name, mode_id in ALL_MODES.items():
            mode_ids = torch.full((n_sample, 1), mode_id, dtype=torch.long, device=ctx.device)
            actions = ctx.biojepa.composer.encode_sequence_path(seq_emb, modality_ids, mode_ids, pert_mask)
            mode_actions[mode_name] = actions.squeeze(1).cpu().numpy()

    pairwise_distances = {}
    mode_list = list(ALL_MODES.keys())
    for i, m1 in enumerate(mode_list):
        for m2 in mode_list[i+1:]:
            dists = [np.linalg.norm(mode_actions[m1][j] - mode_actions[m2][j]) for j in range(n_sample)]
            pairwise_distances[f'{m1}_vs_{m2}'] = {'mean': float(np.mean(dists)), 'std': float(np.std(dists))}

    all_actions = np.concatenate([mode_actions[m] for m in ALL_MODES.keys()], axis=0)
    all_labels = np.array([i for i, m in enumerate(ALL_MODES.keys()) for _ in range(n_sample)])

    train_idx, val_idx = train_test_split(np.arange(len(all_actions)), test_size=0.3, random_state=42, stratify=all_labels)
    _, _, mode_acc = train_linear_classifier(all_actions[train_idx], all_labels[train_idx], all_actions[val_idx], all_labels[val_idx], len(ALL_MODES), ctx.device, epochs=100)

    chance = 1.0 / len(ALL_MODES)
    print(f'mode_sensitivity: Classification_acc={mode_acc:.4f} ({mode_acc/chance:.1f}x chance)')

    return {
        'config': {'n_samples': n_sample, 'modes': list(ALL_MODES.keys())},
        'pairwise_distances': pairwise_distances,
        'classification': {'accuracy': float(mode_acc), 'chance': float(chance), 'above_chance_ratio': float(mode_acc / chance)}
    }


def _fusion_quality(ctx):
    '''Does fusion improve over sequence-only or target-only paths?'''
    pairs = ctx.alignment_pairs
    inf = ctx.alignment_inference
    if 'dna_actions' not in inf or 'target_actions' not in inf:
        return {'error': 'Need both sequence and target embeddings'}

    dna_actions = inf['dna_actions']
    target_actions = inf['target_actions']

    dna_mask = pairs['modality'] == 0
    if dna_mask.sum() == 0:
        return {'error': 'No DNA pairs for fusion test'}

    seq_idx = pairs['seq_idx'][dna_mask][:500]
    target_idx = pairs['target_idx'][dna_mask][:500]
    mode = pairs['mode'][dna_mask][:500]
    n_test = len(seq_idx)

    with torch.no_grad():
        dna_bank = ctx.seq_banks.get('dna') if ctx.seq_banks else None
        target_bank = ctx.target_bank
        if dna_bank is None or target_bank is None:
            return {'error': 'Feature banks not available'}

        seq_emb = torch.zeros(n_test, 1, MAX_SEQ_DIM, device=ctx.device)
        for i, s in enumerate(seq_idx):
            if s < dna_bank.shape[0]:
                emb = dna_bank[s]
                if emb.shape[-1] < MAX_SEQ_DIM:
                    emb = F.pad(emb, (0, MAX_SEQ_DIM - emb.shape[-1]))
                seq_emb[i, 0, :] = emb

        target_emb = torch.zeros(n_test, 1, target_bank.shape[-1], device=ctx.device)
        for i, t in enumerate(target_idx):
            if t < target_bank.shape[0]:
                target_emb[i, 0] = target_bank[t]

        modality_ids = torch.zeros(n_test, 1, dtype=torch.long, device=ctx.device)
        mode_ids = torch.from_numpy(mode).long().to(ctx.device).unsqueeze(1)
        has_seq = torch.ones(n_test, 1, dtype=torch.bool, device=ctx.device)
        has_target = torch.ones(n_test, 1, dtype=torch.bool, device=ctx.device)
        pert_mask = torch.ones(n_test, 1, dtype=torch.bool, device=ctx.device)

        fused_actions = ctx.biojepa.composer(seq_emb, target_emb, modality_ids, mode_ids, has_seq, has_target, pert_mask).squeeze(1)
        seq_only_actions = ctx.biojepa.composer.encode_sequence_path(seq_emb, modality_ids, mode_ids, pert_mask).squeeze(1)
        target_only_actions = ctx.biojepa.composer.encode_target_path(target_emb, mode_ids, pert_mask).squeeze(1)

    fused_np = fused_actions.cpu().numpy()
    seq_np = seq_only_actions.cpu().numpy()
    target_np = target_only_actions.cpu().numpy()

    fused_var = float(np.var(fused_np, axis=0).mean())
    seq_var = float(np.var(seq_np, axis=0).mean())
    target_var = float(np.var(target_np, axis=0).mean())
    fused_norm = F.normalize(fused_actions, dim=1)
    seq_norm = F.normalize(seq_only_actions, dim=1)
    target_norm = F.normalize(target_only_actions, dim=1)
    fused_seq_sim = float(torch.mean(torch.sum(fused_norm * seq_norm, dim=1)).item())
    fused_target_sim = float(torch.mean(torch.sum(fused_norm * target_norm, dim=1)).item())
    seq_target_sim = float(torch.mean(torch.sum(seq_norm * target_norm, dim=1)).item())

    print(f'fusion_quality: Fused_var={fused_var:.4f}, Seq_var={seq_var:.4f}, Target_var={target_var:.4f}')

    return {
        'config': {'n_test_samples': n_test},
        'variance': {'fused': fused_var, 'seq_only': seq_var, 'target_only': target_var},
        'similarity': {'fused_to_seq': fused_seq_sim, 'fused_to_target': fused_target_sim, 'seq_to_target': seq_target_sim}
    }


def _missing_data_robustness(ctx):
    '''How gracefully does model degrade with missing information?'''
    pairs = ctx.alignment_pairs
    inf = ctx.alignment_inference
    if 'target_actions_norm' not in inf:
        return {'error': 'No target embeddings available for robustness test'}

    target_norm = inf['target_actions_norm']
    dna_bank = ctx.seq_banks.get('dna') if ctx.seq_banks else None
    target_bank = ctx.target_bank
    if dna_bank is None or target_bank is None:
        return {'error': 'Feature banks not available'}

    dna_mask = pairs['modality'] == 0
    seq_idx = pairs['seq_idx'][dna_mask][:200]
    target_idx = pairs['target_idx'][dna_mask][:200]
    mode = pairs['mode'][dna_mask][:200]
    n_test = len(seq_idx)
    if n_test < 10:
        return {'error': 'Not enough test pairs'}

    with torch.no_grad():
        seq_emb = torch.zeros(n_test, 1, MAX_SEQ_DIM, device=ctx.device)
        for i, s in enumerate(seq_idx):
            if s < dna_bank.shape[0]:
                emb = dna_bank[s]
                if emb.shape[-1] < MAX_SEQ_DIM:
                    emb = F.pad(emb, (0, MAX_SEQ_DIM - emb.shape[-1]))
                seq_emb[i, 0, :] = emb

        target_emb = torch.zeros(n_test, 1, target_bank.shape[-1], device=ctx.device)
        for i, t in enumerate(target_idx):
            if t < target_bank.shape[0]:
                target_emb[i, 0] = target_bank[t]

        modality_ids = torch.zeros(n_test, 1, dtype=torch.long, device=ctx.device)
        mode_ids = torch.from_numpy(mode).long().to(ctx.device).unsqueeze(1)
        pert_mask = torch.ones(n_test, 1, dtype=torch.bool, device=ctx.device)

        fused_actions = ctx.biojepa.composer(
            seq_emb, target_emb, modality_ids, mode_ids,
            torch.ones(n_test, 1, dtype=torch.bool, device=ctx.device),
            torch.ones(n_test, 1, dtype=torch.bool, device=ctx.device),
            pert_mask
        ).squeeze(1)

        seq_only_actions = ctx.biojepa.composer(
            seq_emb, torch.zeros_like(target_emb), modality_ids, mode_ids,
            torch.ones(n_test, 1, dtype=torch.bool, device=ctx.device),
            torch.zeros(n_test, 1, dtype=torch.bool, device=ctx.device),
            pert_mask
        ).squeeze(1)

        target_only_actions = ctx.biojepa.composer(
            torch.zeros_like(seq_emb), target_emb, modality_ids, mode_ids,
            torch.zeros(n_test, 1, dtype=torch.bool, device=ctx.device),
            torch.ones(n_test, 1, dtype=torch.bool, device=ctx.device),
            pert_mask
        ).squeeze(1)

    def retrieval_mrr(query_actions, target_bank_norm, correct_targets):
        query_norm = F.normalize(query_actions, dim=1)
        sim_matrix = torch.mm(query_norm, target_bank_norm.T).cpu().numpy()
        reciprocal_ranks = []
        for i, t in enumerate(correct_targets):
            if t >= sim_matrix.shape[1]:
                continue
            sims = sim_matrix[i]
            sorted_idx = np.argsort(sims)[::-1]
            rank = np.where(sorted_idx == t)[0][0] + 1
            reciprocal_ranks.append(1.0 / rank)
        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    fused_mrr = retrieval_mrr(fused_actions, target_norm, target_idx)
    seq_only_mrr = retrieval_mrr(seq_only_actions, target_norm, target_idx)
    target_only_mrr = retrieval_mrr(target_only_actions, target_norm, target_idx)

    seq_recovery = seq_only_mrr / fused_mrr if fused_mrr > 0 else 0.0
    target_recovery = target_only_mrr / fused_mrr if fused_mrr > 0 else 0.0

    print(f'missing_data_robustness: Fused_MRR={fused_mrr:.4f}, Seq_only={seq_only_mrr:.4f}, Target_only={target_only_mrr:.4f}')

    return {
        'config': {'n_test_samples': n_test},
        'mrr': {'fused': fused_mrr, 'seq_only': seq_only_mrr, 'target_only': target_only_mrr},
        'recovery_ratio': {'seq_only_vs_fused': float(seq_recovery), 'target_only_vs_fused': float(target_recovery)}
    }


def _multi_pert_alignment(ctx):
    '''Multi-perturbation alignment using real Norman dual-gene samples.'''
    if ctx.seq_banks is None or 'dna' not in ctx.seq_banks or ctx.target_bank is None:
        return {'error': 'Feature banks not available'}

    dna_bank = ctx.seq_banks['dna']
    target_bank = ctx.target_bank

    try:
        test_loader = TrainingLoader(batch_size=ctx.config['batch_size'], split='test', data_dir=ctx.paths['train_dir'], device=ctx.device)
    except RuntimeError as e:
        return {'error': f'Could not load test shards: {e}. Run data_prep_03_shards.ipynb first.'}

    test_steps = min(500, ctx.config['test_total_examples'] // ctx.config['batch_size'])
    single_pert_samples, multi_pert_samples = [], []

    for _ in tqdm(range(test_steps), desc='multi_pert_alignment: Scanning for multi-pert'):
        batch = test_loader.next_batch()
        for i in range(batch.n_perts.shape[0]):
            n = int(batch.n_perts[i].item())
            if batch.modality[i, 0].item() != 0:
                continue
            if n == 1:
                s, t, m = int(batch.seq_idx[i, 0]), int(batch.target_idx[i, 0]), int(batch.mode[i, 0])
                if s >= 0 and s < dna_bank.shape[0] and t >= 0 and t < target_bank.shape[0]:
                    single_pert_samples.append((s, t, m))
            elif n > 1:
                perts = []
                valid = True
                for j in range(n):
                    s, t, m = int(batch.seq_idx[i, j]), int(batch.target_idx[i, j]), int(batch.mode[i, j])
                    if s < 0 or s >= dna_bank.shape[0] or t < 0 or t >= target_bank.shape[0]:
                        valid = False
                        break
                    perts.append((s, t, m))
                if valid:
                    multi_pert_samples.append(perts)

    if len(multi_pert_samples) < 10:
        return {'error': f'Not enough multi-pert samples (found {len(multi_pert_samples)}). Norman dual-gene may not be in test split.'}

    if len(single_pert_samples) < 10:
        return {'error': f'Not enough single-pert samples for comparison (found {len(single_pert_samples)})'}

    rng = np.random.RandomState(42)
    single_sample = rng.choice(len(single_pert_samples), min(200, len(single_pert_samples)), replace=False)
    single_sims = []

    with torch.no_grad():
        for idx in single_sample:
            s, t, m = single_pert_samples[idx]
            seq_emb = torch.zeros(1, 1, MAX_SEQ_DIM, device=ctx.device)
            emb = dna_bank[s]
            if emb.shape[-1] < MAX_SEQ_DIM:
                emb = F.pad(emb, (0, MAX_SEQ_DIM - emb.shape[-1]))
            seq_emb[0, 0] = emb
            target_emb = target_bank[t].unsqueeze(0).unsqueeze(0)
            mod = torch.zeros(1, 1, dtype=torch.long, device=ctx.device)
            mode = torch.full((1, 1), m, dtype=torch.long, device=ctx.device)
            mask = torch.ones(1, 1, dtype=torch.bool, device=ctx.device)
            seq_action = ctx.biojepa.composer.encode_sequence_path(seq_emb, mod, mode, mask)
            target_action = ctx.biojepa.composer.encode_target_path(target_emb, mode, mask)
            seq_pooled = ctx.biojepa.composer.attention_pool(seq_action, mask)
            target_pooled = ctx.biojepa.composer.attention_pool(target_action, mask)
            sim = F.cosine_similarity(seq_pooled, target_pooled, dim=1).item()
            single_sims.append(sim)

    multi_sample = rng.choice(len(multi_pert_samples), min(100, len(multi_pert_samples)), replace=False)
    multi_sims = []

    with torch.no_grad():
        for idx in multi_sample:
            perts = multi_pert_samples[idx]
            n = len(perts)
            seq_emb = torch.zeros(1, n, MAX_SEQ_DIM, device=ctx.device)
            target_emb = torch.zeros(1, n, target_bank.shape[-1], device=ctx.device)
            mode_ids = torch.zeros(1, n, dtype=torch.long, device=ctx.device)
            for j, (s, t, m) in enumerate(perts):
                emb = dna_bank[s]
                if emb.shape[-1] < MAX_SEQ_DIM:
                    emb = F.pad(emb, (0, MAX_SEQ_DIM - emb.shape[-1]))
                seq_emb[0, j] = emb
                target_emb[0, j] = target_bank[t]
                mode_ids[0, j] = m
            mod = torch.zeros(1, n, dtype=torch.long, device=ctx.device)
            mask = torch.ones(1, n, dtype=torch.bool, device=ctx.device)
            seq_action = ctx.biojepa.composer.encode_sequence_path(seq_emb, mod, mode_ids, mask)
            target_action = ctx.biojepa.composer.encode_target_path(target_emb, mode_ids, mask)
            seq_pooled = ctx.biojepa.composer.attention_pool(seq_action, mask)
            target_pooled = ctx.biojepa.composer.attention_pool(target_action, mask)
            sim = F.cosine_similarity(seq_pooled, target_pooled, dim=1).item()
            multi_sims.append(sim)

    single_mean = float(np.mean(single_sims))
    multi_mean = float(np.mean(multi_sims))
    degradation = 1.0 - (multi_mean / single_mean) if single_mean > 0 else 0.0

    print(f'multi_pert_alignment: Single={single_mean:.4f}, Multi={multi_mean:.4f}, Degradation={degradation:.2%}')

    return {
        'config': {'n_single_samples': len(single_sims), 'n_multi_samples': len(multi_sims), 'n_multi_pert_total': len(multi_pert_samples)},
        'single_pert': {'mean_sim': single_mean, 'std': float(np.std(single_sims))},
        'multi_pert': {'mean_sim': multi_mean, 'std': float(np.std(multi_sims))},
        'degradation': float(degradation)
    }


def _target_family_probing(ctx):
    '''Do action embeddings encode protein family information? Tests seq-only, target-only, and fused.'''
    inf = ctx.alignment_inference
    gene_families = ctx.hgnc_gene_families['by_symbol']

    id_mapping_path = ctx.paths['pert_dir'] / 'input_to_id.json'
    if id_mapping_path.exists():
        with open(id_mapping_path) as f:
            input_to_id = json.load(f)
        id_to_gene = {pid: key.split('_')[0].upper() for key, pid in input_to_id.items()}
    else:
        idx_mapping_path = ctx.paths['pert_dir'] / 'seq_banks' / 'dna_to_idx.json'
        if idx_mapping_path.exists():
            with open(idx_mapping_path) as f:
                dna_to_idx = json.load(f)
            id_to_gene = {idx: key.split('_')[0].upper() for key, idx in dna_to_idx.items()}
        else:
            return {'error': 'No gene mapping file found'}

    pert_families = {pid: gene_families[gene] for pid, gene in id_to_gene.items() if gene in gene_families}

    family_counts = defaultdict(int)
    for fam in pert_families.values():
        family_counts[fam] += 1
    valid_families = {fam for fam, count in family_counts.items() if count >= 5}
    valid_perts = [pid for pid, fam in pert_families.items() if fam in valid_families]

    if len(valid_perts) < 20:
        return {'error': f'Not enough perturbations with valid families (found {len(valid_perts)})'}

    family_to_idx = {fam: i for i, fam in enumerate(sorted(valid_families))}
    n_families = len(family_to_idx)
    labels = np.array([family_to_idx[pert_families[pid]] for pid in valid_perts])

    results = {}

    if 'dna_actions' in inf:
        dna_actions = inf['dna_actions'].cpu().numpy()
        valid_perts_in_range = [p for p in valid_perts if p < dna_actions.shape[0]]
        if len(valid_perts_in_range) >= 20:
            X = dna_actions[valid_perts_in_range]
            y = np.array([family_to_idx[pert_families[pid]] for pid in valid_perts_in_range])
            train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.3, random_state=42, stratify=y)
            _, val_preds, val_acc = train_linear_classifier(X[train_idx], y[train_idx], X[val_idx], y[val_idx], n_families, ctx.device, epochs=200)
            results['seq_only'] = {
                'accuracy': float(val_acc),
                'macro_f1': float(f1_score(y[val_idx], val_preds, average='macro')),
                'n_samples': len(valid_perts_in_range)
            }

    if 'target_actions' in inf:
        pairs = ctx.alignment_pairs
        dna_mask = pairs['modality'] == 0
        seq_to_target = {}
        for s, t in zip(pairs['seq_idx'][dna_mask], pairs['target_idx'][dna_mask]):
            seq_to_target[s] = t

        target_actions = inf['target_actions'].cpu().numpy()
        valid_perts_with_target = [p for p in valid_perts if p in seq_to_target and seq_to_target[p] < target_actions.shape[0]]
        if len(valid_perts_with_target) >= 20:
            X = np.array([target_actions[seq_to_target[p]] for p in valid_perts_with_target])
            y = np.array([family_to_idx[pert_families[pid]] for pid in valid_perts_with_target])
            train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.3, random_state=42, stratify=y)
            _, val_preds, val_acc = train_linear_classifier(X[train_idx], y[train_idx], X[val_idx], y[val_idx], n_families, ctx.device, epochs=200)
            results['target_only'] = {
                'accuracy': float(val_acc),
                'macro_f1': float(f1_score(y[val_idx], val_preds, average='macro')),
                'n_samples': len(valid_perts_with_target)
            }

    if ctx.seq_banks and 'dna' in ctx.seq_banks and ctx.target_bank is not None:
        dna_bank = ctx.seq_banks['dna']
        target_bank_tensor = ctx.target_bank
        pairs = ctx.alignment_pairs
        dna_mask = pairs['modality'] == 0
        seq_to_target = {int(s): int(t) for s, t in zip(pairs['seq_idx'][dna_mask], pairs['target_idx'][dna_mask])}
        valid_fused = [p for p in valid_perts if p < dna_bank.shape[0] and p in seq_to_target and seq_to_target[p] < target_bank_tensor.shape[0]]
        if len(valid_fused) >= 20:
            n_test = len(valid_fused)
            with torch.no_grad():
                seq_emb = torch.zeros(n_test, 1, MAX_SEQ_DIM, device=ctx.device)
                target_emb = torch.zeros(n_test, 1, target_bank_tensor.shape[-1], device=ctx.device)
                for i, p in enumerate(valid_fused):
                    emb = dna_bank[p]
                    if emb.shape[-1] < MAX_SEQ_DIM:
                        emb = F.pad(emb, (0, MAX_SEQ_DIM - emb.shape[-1]))
                    seq_emb[i, 0] = emb
                    target_emb[i, 0] = target_bank_tensor[seq_to_target[p]]
                modality_ids = torch.zeros(n_test, 1, dtype=torch.long, device=ctx.device)
                mode_ids = torch.zeros(n_test, 1, dtype=torch.long, device=ctx.device)
                has_seq = torch.ones(n_test, 1, dtype=torch.bool, device=ctx.device)
                has_target = torch.ones(n_test, 1, dtype=torch.bool, device=ctx.device)
                pert_mask = torch.ones(n_test, 1, dtype=torch.bool, device=ctx.device)
                fused_actions = ctx.biojepa.composer(seq_emb, target_emb, modality_ids, mode_ids, has_seq, has_target, pert_mask).squeeze(1)
            X = fused_actions.cpu().numpy()
            y = np.array([family_to_idx[pert_families[pid]] for pid in valid_fused])
            train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.3, random_state=42, stratify=y)
            _, val_preds, val_acc = train_linear_classifier(X[train_idx], y[train_idx], X[val_idx], y[val_idx], n_families, ctx.device, epochs=200)
            results['fused'] = {
                'accuracy': float(val_acc),
                'macro_f1': float(f1_score(y[val_idx], val_preds, average='macro')),
                'n_samples': len(valid_fused)
            }

    chance = 1.0 / n_families
    for key in results:
        results[key]['chance'] = chance
        results[key]['above_chance_ratio'] = results[key]['accuracy'] / chance

    acc_summary = ', '.join(f'{k}={v["accuracy"]:.4f}' for k, v in results.items())
    print(f'target_family_probing: {acc_summary}')

    return {'config': {'n_families': n_families, 'n_valid_perts': len(valid_perts)}, 'by_embedding_type': results}
