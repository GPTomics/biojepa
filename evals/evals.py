'''BioJEPA evaluation utilities and suite entry points.'''

import json
import pickle
import random
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

import biojepa_v1_0 as model
from dataloader_v1_0 import TrainingLoader, EvalLoader
from .linear_expression_decoder import BenchmarkDecoder, BenchmarkDecoderConfig
from .pathway_utils import load_pathway_annotations, build_gene_to_pathways, compute_multilabel_pathway_similarity
from .linear_classifier import train_linear_classifier
import torch.nn.functional as F
from config_v1_0 import MAX_SEQ_DIM, VERSION


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
HEAVY_INFERENCE_KEYS = {'sample_pred_deltas', 'sample_real_deltas', 'sample_logvars'}


class _RunningMeans:
    def __init__(self, n_genes, embed_dim=0):
        z = lambda: np.zeros(n_genes, dtype=np.float64)
        self._sums = defaultdict(lambda: {'pd': z(), 'rd': z(), 'pa': z(), 'ra': z(), 'ctrl': z()})
        self._gene_counts = defaultdict(lambda: np.zeros(n_genes, dtype=np.int64))
        self._sample_counts = defaultdict(int)
        self._embed_dim = embed_dim
        if embed_dim > 0:
            self._latent_sums = defaultdict(lambda: np.zeros(embed_dim, dtype=np.float64))

    def add(self, key, pred_delta, real_delta, pred_abs, real_abs, control, gene_mask=None, latent_delta=None):
        s = self._sums[key]
        if gene_mask is not None:
            s['pd'] += np.where(gene_mask, pred_delta, 0.0)
            s['rd'] += np.where(gene_mask, real_delta, 0.0)
            s['pa'] += np.where(gene_mask, pred_abs, 0.0)
            s['ra'] += np.where(gene_mask, real_abs, 0.0)
            s['ctrl'] += np.where(gene_mask, control, 0.0)
            self._gene_counts[key] += gene_mask.astype(np.int64)
        else:
            s['pd'] += pred_delta
            s['rd'] += real_delta
            s['pa'] += pred_abs
            s['ra'] += real_abs
            s['ctrl'] += control
            self._gene_counts[key] += 1
        self._sample_counts[key] += 1
        if latent_delta is not None and self._embed_dim > 0:
            self._latent_sums[key] += latent_delta

    def finalize(self):
        keys = list(self._sums.keys())
        def _gene_mean(field):
            return {k: (self._sums[k][field] / np.maximum(self._gene_counts[k], 1)).astype(np.float32) for k in keys}
        result = {
            'mean_pred_deltas': _gene_mean('pd'), 'mean_real_deltas': _gene_mean('rd'),
            'mean_pred_abs': _gene_mean('pa'), 'mean_real_abs': _gene_mean('ra'),
            'mean_control_states': _gene_mean('ctrl'),
        }
        if self._embed_dim > 0:
            result['mean_latent_deltas'] = {k: (self._latent_sums[k] / self._sample_counts[k]).astype(np.float32) for k in keys}
        return keys, result

    def get_gene_masks(self):
        return {k: self._gene_counts[k] > 0 for k in self._sums.keys()}


def _build_multi_pert_key(all_target_idx, all_seq_idx, all_modality, all_mode, n_perts, cell_type):
    slot_ids = []
    for j in range(n_perts):
        t, s = int(all_target_idx[j]), int(all_seq_idx[j])
        mod, mode_val = int(all_modality[j]), int(all_mode[j])
        slot_ids.append(('t', t, mod, mode_val) if t >= 0 else ('s', s, mod, mode_val))
    return (tuple(sorted(slot_ids)), int(cell_type))


def _build_sample_gene_masks(inf, dataset_gene_masks):
    if not dataset_gene_masks:
        return None
    ds_ids = inf.get('sample_dataset_ids')
    ds_id_to_name = inf.get('dataset_id_to_name', {})
    if ds_ids is None:
        return None
    n_genes = len(next(iter(dataset_gene_masks.values())))
    masks = np.ones((len(ds_ids), n_genes), dtype=np.bool_)
    for ds_id, ds_name in ds_id_to_name.items():
        if ds_name in dataset_gene_masks:
            masks[ds_ids == ds_id] = dataset_gene_masks[ds_name]
    return masks


class _LazyNpzDict(dict):
    '''Dict that lazy-loads heavy arrays from shard .npz files on first access.'''
    def __init__(self, data, shard_paths):
        super().__init__(data)
        self._shard_paths = shard_paths

    def __getitem__(self, key):
        if key in HEAVY_INFERENCE_KEYS and key not in dict.keys(self):
            arrays = []
            for path in self._shard_paths:
                npz = np.load(path)
                arrays.append(npz[key])
                npz.close()
            dict.__setitem__(self, key, np.concatenate(arrays, axis=0))
        return dict.__getitem__(self, key)


class EvalContext:
    '''Unified context for running evaluations. All loading is lazy.'''

    def __init__(self, config, data_root, checkpoint_root, ref_dir):
        missing_keys = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
        if missing_keys:
            raise ValueError(f'Missing required config keys: {missing_keys}. Required: {REQUIRED_CONFIG_KEYS}')
        self.config = {
            'pert_latent_dim': 128,
            'pert_mode_dim': 64,
            'predictor_embed_dim': 128,
            'predictor_n_layer': 4,
            'predictor_heads': 4,
            'verbose': True,
            **config,
        }
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.paths = self._get_paths(Path(data_root), Path(checkpoint_root), Path(ref_dir))
        if self.config['verbose']:
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
        self._norman_single_gene_deltas = None
        self._norman_combo_mapping = None
        self._norman_gi_subtypes = None
        self._dataset_splits = None
        self._dataset_gene_masks = None

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
            'train_dir': data_root / 'predictor_t',
            'checkpoint_dir': checkpoint_root / 'checkpoint',
            'pert_dir': data_root / 'pert_embd',
            'ref_dir': ref_dir
        }

    @property
    def biojepa(self):
        if self._biojepa is None:
            if 'checkpoint_path' not in self.config:
                raise ValueError('checkpoint_path required in config to load model from disk. '
                                 'For mid-training evals, inject model via eval_ctx._biojepa = model instead.')
            print('Loading BioJEPA model...')
            torch.set_float32_matmul_precision('high')
            model_config = model.BioJepaConfig(
                num_genes=self.config['num_genes'],
                n_layer=self.config['n_layer'],
                heads=self.config['heads'],
                embed_dim=self.config['embed_dim'],
                n_pre_layer=self.config.get('n_pre_layer', self.config['n_layer']),
                mlp_ratio=self.config.get('mlp_ratio', 4.0),
                mask_ratio=self.config.get('mask_ratio', 0.6),
                gaussian_scale=self.config.get('gaussian_scale', 2.0),
                film_linear_multiple=self.config.get('film_linear_multiple', 1.0),
                sim_coeff=self.config.get('sim_coeff', 25.0),
                std_coeff=self.config.get('std_coeff', 25.0),
                cov_coeff=self.config.get('cov_coeff', 1.0),
                pert_latent_dim=self.config.get('pert_latent_dim', 128),
                pert_mode_dim=self.config.get('pert_mode_dim', 64),
                predictor_embed_dim=self.config.get('predictor_embed_dim', 128),
                predictor_n_layer=self.config.get('predictor_n_layer', 4),
                predictor_heads=self.config.get('predictor_heads', 4),
                ema_momentum=self.config.get('ema_momentum', 0.995),
            )
            self._biojepa = model.BioJepa(model_config).to(self.device)
            checkpoint_path = Path(self.config['checkpoint_path'])
            if not checkpoint_path.is_absolute():
                checkpoint_path = self.paths['checkpoint_dir'] / checkpoint_path
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint['model']
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            print(self._biojepa.load_state_dict(state_dict))
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
            checkpoint = torch.load(self.paths['checkpoint_dir'] / f'biojepa_{VERSION}_decoder_final.pt', map_location=self.device)
            self._decoder.load_state_dict(checkpoint['model'])
            self._decoder.eval()
        return self._decoder

    @property
    def gene_embeddings(self):
        if self._gene_embeddings is None:
            self._gene_embeddings = self.biojepa.teacher.gene_embeddings.detach().cpu().numpy()
        return self._gene_embeddings

    @property
    def gene_names(self):
        if self._gene_names is None:
            with open(self.paths['data_dir'] / 'gene_names.json') as f:
                self._gene_names = json.load(f)
        return self._gene_names

    @property
    def seq_banks(self):
        '''Load sequence embedding banks (DNA, chemical) for v1.0 dual-path alignment.'''
        if self._seq_banks is None:
            seq_banks_dir = self.paths['pert_dir'] / 'seq_banks'
            self._seq_banks = {}
            dna_path = seq_banks_dir / 'dna_embeddings.npy'
            if dna_path.exists():
                self._seq_banks['dna'] = torch.from_numpy(np.load(dna_path)).float().to(self.device)
                print(f'Loaded DNA seq bank: {self._seq_banks["dna"].shape}')
            chem_path = seq_banks_dir / 'chemical_embeddings.npy'
            if chem_path.exists():
                chem = torch.from_numpy(np.load(chem_path)).float().to(self.device)
                if chem.shape[-1] < MAX_SEQ_DIM:
                    print(f'Warning: Chemical embeddings need padding ({chem.shape[-1]} -> {MAX_SEQ_DIM}). '
                          f'Run data_prep/prepad_embeddings.py for permanent fix.')
                    chem = F.pad(chem, (0, MAX_SEQ_DIM - chem.shape[-1]))
                self._seq_banks['chemical'] = chem
                print(f'Loaded chemical seq bank: {self._seq_banks["chemical"].shape}')
        return self._seq_banks

    @property
    def target_bank(self):
        '''Load protein target embedding bank for v1.0 dual-path alignment.'''
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
                print(f'Loaded {len(self._alignment_pairs["seq_idx"])} v1.0 alignment pairs')
            elif old_path.exists():
                with np.load(old_path) as data:
                    n_pairs = len(data['input_idx'])
                    self._alignment_pairs = {
                        'seq_idx': data['input_idx'],
                        'target_idx': data['anchor_idx'],
                        'modality': np.zeros(n_pairs, dtype=np.int64),
                        'mode': np.zeros(n_pairs, dtype=np.int64)
                    }
                print(f'Loaded {n_pairs} v0.5 alignment pairs (converted to v1.0 format)')
            else:
                raise FileNotFoundError(f'No alignment pairs found at {new_path} or {old_path}')
        return self._alignment_pairs

    @property
    def alignment_inference(self):
        '''Cached action vectors for alignment evals using v1.0 dual-path architecture.

        Uses encode_sequence_path() for sequences and encode_target_path() for targets.
        Supports DNA and chemical modalities for sequences, protein targets only.
        '''
        if self._alignment_inference is None:
            result = {}

            with torch.no_grad():
                if self.seq_banks and 'dna' in self.seq_banks:
                    dna_emb = self.seq_banks['dna']
                    n_dna = dna_emb.shape[0]
                    seq_emb = dna_emb.unsqueeze(1)
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
                    seq_emb = chem_emb.unsqueeze(1)
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
            self._pathway_annotations = load_pathway_annotations(['KEGG_2026', 'Reactome_Pathways_2024'])
        return self._pathway_annotations

    @property
    def id_to_gene(self):
        '''Cached mapping from DNA perturbation seq_idx to gene symbol.'''
        if self._id_to_gene is None:
            dna_path = self.paths['pert_dir'] / 'seq_banks' / 'dna_to_idx.json'
            with open(dna_path) as f:
                dna_to_idx = json.load(f)
            self._id_to_gene = {idx: key.split('_')[0].upper() for key, idx in dna_to_idx.items()}
        return self._id_to_gene

    @property
    def norman_single_gene_deltas(self):
        if self._norman_single_gene_deltas is None:
            path = self.paths['data_dir'] / 'norman_single_gene_deltas.npz'
            if path.exists():
                data = np.load(path)
                self._norman_single_gene_deltas = {
                    'gene_names': list(data['gene_names']),
                    'deltas': data['deltas'],
                    'mean_control': data['mean_control'],
                }
                if self.config['verbose']:
                    print(f'Loaded Norman single-gene deltas: {len(self._norman_single_gene_deltas["gene_names"])} genes')
        return self._norman_single_gene_deltas

    @property
    def norman_combo_mapping(self):
        if self._norman_combo_mapping is None:
            path = self.paths['data_dir'] / 'norman_combo_mapping.json'
            if path.exists():
                with open(path) as f:
                    self._norman_combo_mapping = json.load(f)
                if self.config['verbose']:
                    print(f'Loaded Norman combo mapping: {len(self._norman_combo_mapping)} combos')
        return self._norman_combo_mapping

    @property
    def norman_gi_subtypes(self):
        if self._norman_gi_subtypes is None:
            path = self.paths['ref_dir'] / 'norman' / 'norman_gi_subtypes.json'
            if path.exists():
                with open(path) as f:
                    self._norman_gi_subtypes = json.load(f)
                if self.config['verbose']:
                    print(f'Loaded Norman GI subtypes: {len(self._norman_gi_subtypes)} combos')
        return self._norman_gi_subtypes

    @property
    def dataset_splits(self):
        if self._dataset_splits is None:
            path = self.paths['data_dir'] / 'dataset_splits.json'
            if path.exists():
                with open(path) as f:
                    self._dataset_splits = json.load(f)
                if self.config['verbose']:
                    print(f'Loaded dataset splits: {list(self._dataset_splits.keys())}')
        return self._dataset_splits

    @property
    def dataset_gene_masks(self):
        if self._dataset_gene_masks is None:
            path = self.paths['data_dir'] / 'dataset_gene_masks.json'
            if path.exists():
                with open(path) as f:
                    self._dataset_gene_masks = {k: np.array(v, dtype=np.bool_) for k, v in json.load(f).items()}
                if self.config['verbose']:
                    print(f'Loaded dataset gene masks: {list(self._dataset_gene_masks.keys())}')
        return self._dataset_gene_masks

    @property
    def test_inference(self):
        if self._test_inference is None:
            if not self._load_inference_cache():
                inf = self._run_test_inference()
                self._save_inference_cache(inf)
        return self._test_inference

    def _save_inference_cache(self, inf):
        cache_dir = self.paths['data_dir'] / 'test_inference_cache'
        with open(cache_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(inf, f)
        shard_paths = sorted(cache_dir.glob('shard_*.npz'))
        self._test_inference = _LazyNpzDict(inf, shard_paths)
        print(f'Cached test inference to {cache_dir} ({len(shard_paths)} shards)')

    def _load_inference_cache(self):
        cache_dir = self.paths['data_dir'] / 'test_inference_cache'
        meta_path = cache_dir / 'metadata.pkl'
        if not meta_path.exists():
            return False
        shard_paths = sorted(cache_dir.glob('shard_*.npz'))
        if not shard_paths:
            return False
        with open(meta_path, 'rb') as f:
            lightweight = pickle.load(f)
        self._test_inference = _LazyNpzDict(lightweight, shard_paths)
        print(f'Loaded cached test inference from {cache_dir} ({len(shard_paths)} shards, delete directory to recompute)')
        return True

    def _run_test_inference(self):
        '''Run inference on test set. Returns aggregated and per-sample data.'''
        test_loader = EvalLoader(batch_size=self.config['batch_size'], split=self.config.get('eval_split', 'test'), data_dir=self.paths['train_dir'], device=self.device, seed=self.config.get('seed', 1337))
        test_steps = self.config.get('test_total_examples', test_loader.total_samples) // self.config['batch_size']
        N = self.config['num_genes']

        shard_size = self.config.get('inference_shard_size', 2560)
        cache_dir = self.paths['data_dir'] / 'test_inference_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        for old_shard in cache_dir.glob('shard_*.npz'):
            old_shard.unlink()

        shard_pred, shard_real, shard_logvar = [], [], []
        shard_samples, shard_idx = 0, 0

        def _flush_shard():
            nonlocal shard_pred, shard_real, shard_logvar, shard_samples, shard_idx
            if not shard_pred:
                return
            np.savez_compressed(
                cache_dir / f'shard_{shard_idx:04d}.npz',
                sample_pred_deltas=np.concatenate(shard_pred),
                sample_real_deltas=np.concatenate(shard_real),
                sample_logvars=np.concatenate(shard_logvar),
            )
            shard_pred, shard_real, shard_logvar = [], [], []
            shard_samples = 0
            shard_idx += 1

        E = self.config['embed_dim']
        bulk_global = _RunningMeans(N, E)
        multi_pert_global = _RunningMeans(N, E)
        multi_pert_first_seq_idx = {}
        ds_running = defaultdict(lambda: _RunningMeans(N, E))
        ct_running = defaultdict(lambda: _RunningMeans(N, E))

        sample_pert_ids, sample_target_ids, sample_pert_mods = [], [], []
        sample_mses, sample_correlations = [], []
        sample_n_perts, sample_cell_types, sample_doses = [], [], []
        sample_all_seq_idx, sample_all_target_idx = [], []
        sample_all_modality, sample_all_mode = [], []
        sample_dataset_ids = []
        all_control_totals = []
        ds_sample_mses, ds_sample_correlations = defaultdict(list), defaultdict(list)
        ct_sample_mses, ct_sample_correlations = defaultdict(list), defaultdict(list)

        for _ in tqdm(range(test_steps), desc='Running test inference', disable=not self.config.get('verbose', True)):
            batch = test_loader.next_batch()
            cont_x, cont_tot = batch.control, batch.control_total
            case_x, case_tot = batch.case, batch.case_total
            all_control_totals.append(cont_tot.cpu().numpy())
            B = cont_x.shape[0]
            N_pert = batch.seq_idx.shape[1]

            pert_mask = torch.arange(N_pert, device=self.device).unsqueeze(0) < batch.n_perts.unsqueeze(1)
            seq_emb = get_seq_embeddings(batch.seq_idx, batch.modality, self.seq_banks)
            target_emb = get_target_embeddings(batch.target_idx, self.target_bank)

            unknown_mask = ~batch.gene_mask if hasattr(batch, 'gene_mask') and batch.gene_mask is not None else None

            with torch.no_grad():
                z_context = self.biojepa.teacher(cont_x, cont_tot, mask_idx=None, unknown_mask=unknown_mask)
                action_latents = self.biojepa.composer(
                    seq_emb, target_emb, batch.modality, batch.mode,
                    batch.has_seq, batch.has_target, pert_mask, dose=batch.dose
                )
                target_indices = torch.arange(N, device=self.device).expand(B, N)
                z_pred_mu, z_pred_logvar = self.biojepa.predictor(z_context, action_latents, target_indices)
                pred_delta = self.decoder(z_pred_mu) - self.decoder(z_context)
                real_delta = case_x - cont_x
                pred_abs = torch.clamp(cont_x + pred_delta, min=0.0)

            z_latent_delta = (z_pred_mu - z_context).mean(dim=1).cpu().numpy()

            pred_delta_np, real_delta_np = pred_delta.cpu().numpy(), real_delta.cpu().numpy()
            pred_abs_np, real_abs_np = pred_abs.cpu().numpy(), case_x.cpu().numpy()
            logvar_np = z_pred_logvar.mean(dim=-1).cpu().numpy()
            p_idx_np = batch.seq_idx[:, 0].cpu().numpy()
            p_target_np = batch.target_idx[:, 0].cpu().numpy()
            p_mod_np = batch.modality[:, 0].cpu().numpy()
            p_mode_np = batch.mode[:, 0].cpu().numpy()
            cont_x_np = cont_x.cpu().numpy()
            ds_id_np = batch.dataset_id.cpu().numpy()
            n_perts_np = batch.n_perts.cpu().numpy()
            cell_type_np = batch.cell_type.cpu().numpy()
            dose_np = batch.dose.cpu().numpy()
            gene_mask_np = batch.gene_mask.cpu().numpy().astype(np.bool_) if hasattr(batch, 'gene_mask') and batch.gene_mask is not None else None
            all_seq_idx_np = batch.seq_idx.cpu().numpy()
            all_target_idx_np = batch.target_idx.cpu().numpy()
            all_modality_np = batch.modality.cpu().numpy()
            all_mode_np = batch.mode.cpu().numpy()

            shard_pred.append(pred_delta_np)
            shard_real.append(real_delta_np)
            shard_logvar.append(logvar_np)
            shard_samples += B

            sample_pert_ids.append(p_idx_np)
            sample_target_ids.append(p_target_np)
            sample_pert_mods.append(p_mod_np)
            sample_n_perts.append(n_perts_np)
            sample_cell_types.append(cell_type_np)
            sample_doses.append(dose_np)
            sample_all_seq_idx.append(all_seq_idx_np)
            sample_all_target_idx.append(all_target_idx_np)
            sample_all_modality.append(all_modality_np)
            sample_all_mode.append(all_mode_np)
            sample_dataset_ids.append(ds_id_np)

            for i in range(B):
                key = (int(p_idx_np[i]), int(p_target_np[i]), int(p_mod_np[i]), int(p_mode_np[i]), int(cell_type_np[i]))

                gm_i = gene_mask_np[i] if gene_mask_np is not None else None
                sample_mse = float(np.mean((pred_delta_np[i][gm_i] - real_delta_np[i][gm_i])**2)) if gm_i is not None else float(np.mean((pred_delta_np[i] - real_delta_np[i])**2))
                if gm_i is not None:
                    measured = np.where(gm_i)[0]
                    k = min(20, len(measured))
                    top_20_idx = measured[np.argsort(np.abs(real_delta_np[i][measured]))[-k:]] if k > 0 else measured
                else:
                    top_20_idx = np.argsort(np.abs(real_delta_np[i]))[-20:]
                p_top, t_top = pred_delta_np[i][top_20_idx], real_delta_np[i][top_20_idx]
                if np.std(p_top) > 1e-9 and np.std(t_top) > 1e-9:
                    corr, _ = pearsonr(p_top, t_top)
                    sample_corr = 0.0 if np.isnan(corr) else float(corr)
                else:
                    sample_corr = 0.0
                sample_mses.append(sample_mse)
                sample_correlations.append(sample_corr)

                ld_i = z_latent_delta[i]
                if n_perts_np[i] > 1:
                    mp_key = _build_multi_pert_key(all_target_idx_np[i], all_seq_idx_np[i], all_modality_np[i], all_mode_np[i], int(n_perts_np[i]), cell_type_np[i])
                    multi_pert_global.add(mp_key, pred_delta_np[i], real_delta_np[i], pred_abs_np[i], real_abs_np[i], cont_x_np[i], gene_mask=gm_i, latent_delta=ld_i)
                    multi_pert_first_seq_idx[mp_key] = int(p_idx_np[i])
                else:
                    bulk_global.add(key, pred_delta_np[i], real_delta_np[i], pred_abs_np[i], real_abs_np[i], cont_x_np[i], gene_mask=gm_i, latent_delta=ld_i)

                ds_name = test_loader.dataset_id_to_name.get(int(ds_id_np[i]), 'unknown')
                if n_perts_np[i] == 1:
                    ds_running[ds_name].add(key, pred_delta_np[i], real_delta_np[i], pred_abs_np[i], real_abs_np[i], cont_x_np[i], gene_mask=gm_i, latent_delta=ld_i)
                ds_sample_mses[ds_name].append(sample_mse)
                ds_sample_correlations[ds_name].append(sample_corr)

                ct = int(cell_type_np[i])
                if n_perts_np[i] == 1:
                    ct_running[ct].add(key, pred_delta_np[i], real_delta_np[i], pred_abs_np[i], real_abs_np[i], cont_x_np[i], gene_mask=gm_i, latent_delta=ld_i)
                ct_sample_mses[ct].append(sample_mse)
                ct_sample_correlations[ct].append(sample_corr)

            if shard_samples >= shard_size:
                _flush_shard()

        _flush_shard()
        pert_keys, bulk_result = bulk_global.finalize()
        pert_gene_masks = bulk_global.get_gene_masks()
        multi_pert_keys, multi_pert_bulk = multi_pert_global.finalize()
        multi_pert_gene_masks = multi_pert_global.get_gene_masks()
        print(f'Aggregated {len(pert_keys)} single-pert, {len(multi_pert_keys)} multi-pert perturbations, {len(sample_mses)} samples, {shard_idx} shards')

        global_mean_control_total = float(np.concatenate(all_control_totals).mean())

        result = {
            'pert_keys': pert_keys, **bulk_result,
            'global_mean_control_total': global_mean_control_total,
            'sample_mses': np.array(sample_mses),
            'sample_correlations': np.array(sample_correlations),
            'sample_pert_ids': np.concatenate(sample_pert_ids, axis=0),
            'sample_target_ids': np.concatenate(sample_target_ids, axis=0),
            'sample_pert_mods': np.concatenate(sample_pert_mods, axis=0),
            'sample_n_perts': np.concatenate(sample_n_perts, axis=0),
            'sample_cell_types': np.concatenate(sample_cell_types, axis=0),
            'sample_doses': np.concatenate(sample_doses, axis=0),
            'sample_all_seq_idx': np.concatenate(sample_all_seq_idx, axis=0),
            'sample_all_target_idx': np.concatenate(sample_all_target_idx, axis=0),
            'sample_all_modality': np.concatenate(sample_all_modality, axis=0),
            'sample_all_mode': np.concatenate(sample_all_mode, axis=0),
            'sample_dataset_ids': np.concatenate(sample_dataset_ids, axis=0),
            'dataset_id_to_name': test_loader.dataset_id_to_name,
            'pert_gene_masks': pert_gene_masks,
        }
        result['multi_pert_keys'] = multi_pert_keys
        result['multi_pert_first_seq_idx'] = multi_pert_first_seq_idx
        result['multi_pert_gene_masks'] = multi_pert_gene_masks
        result.update({f'multi_pert_{k}': v for k, v in multi_pert_bulk.items()})

        by_dataset = {}
        for ds_name in sorted(ds_running.keys()):
            ds_keys, ds_bulk = ds_running[ds_name].finalize()
            ds_pgm = ds_running[ds_name].get_gene_masks()
            by_dataset[ds_name] = {
                'pert_keys': ds_keys, **ds_bulk,
                'pert_gene_masks': ds_pgm,
                'sample_mses': np.array(ds_sample_mses[ds_name]),
                'sample_correlations': np.array(ds_sample_correlations[ds_name]),
            }
            print(f'  {ds_name}: {len(ds_keys)} perturbations, {len(ds_sample_mses[ds_name])} samples')
        result['by_dataset'] = by_dataset

        by_cell_type = {}
        for ct in ct_running:
            ct_keys, ct_bulk = ct_running[ct].finalize()
            ct_pgm = ct_running[ct].get_gene_masks()
            by_cell_type[ct] = {
                'pert_keys': ct_keys, **ct_bulk,
                'pert_gene_masks': ct_pgm,
                'sample_mses': np.array(ct_sample_mses[ct]),
                'sample_correlations': np.array(ct_sample_correlations[ct]),
            }
        result['by_cell_type'] = by_cell_type

        return result


def save_report(results, output_path='eval_report.json'):
    '''Save evaluation results to JSON report.'''
    report_path = Path(output_path)
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        report = {'evals': {}}
    report.pop('version', None)

    run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    report['last_updated'] = run_timestamp
    for name, res in results.items():
        report['evals'][name] = {'run_date': run_timestamp, **res}

    report_path.write_text(json.dumps(report, indent=2))
    print(f'Saved report to {report_path}')


def _permutation_invariance(ctx):
    '''Is the teacher encoder invariant to gene ordering?'''
    verbose = ctx.config['verbose']
    n_permutations = ctx.config.get('n_permutations', 10)
    max_samples = ctx.config.get('perm_inv_max_samples', 64)
    batch_size = ctx.config['batch_size']
    num_genes = ctx.config['num_genes']
    eval_seed = ctx.config.get('seed', 1337)
    split = ctx.config.get('eval_split', 'test')

    shard_dir = ctx.paths['data_dir'] / 'encoder_t' / split
    if not shard_dir.exists():
        return {'skipped': True, 'reason': f'encoder data not found at {shard_dir}'}

    all_shards = sorted(shard_dir.glob('*.npz'))
    shards_by_dataset = defaultdict(list)
    for shard in all_shards:
        parts = shard.stem.split('_')
        if split not in parts:
            continue
        split_idx = parts.index(split)
        if split_idx >= 2:
            shards_by_dataset['_'.join(parts[1:split_idx])].append(shard)

    if not shards_by_dataset:
        return {'skipped': True, 'reason': 'no dataset shards found'}

    rng_py = random.getstate()
    rng_np = np.random.get_state()
    rng_torch = torch.random.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(eval_seed)

    teacher = ctx.biojepa.teacher
    by_dataset = {}
    all_cos_sims = []

    with torch.no_grad():
        for ds_name, ds_shards in sorted(shards_by_dataset.items()):
            with np.load(ds_shards[0]) as data:
                x_np = data['x'].astype(np.float32)
                total_np = data['total'].astype(np.float32)
                gene_mask = data['gene_mask'].astype(np.bool_) if 'gene_mask' in data else np.ones(num_genes, dtype=np.bool_)

            n_use = min(len(x_np), max_samples)
            x = torch.from_numpy(x_np[:n_use]).to(ctx.device)
            total = torch.from_numpy(total_np[:n_use]).to(ctx.device)
            known_mask = torch.from_numpy(gene_mask).to(ctx.device)
            unknown_2d = (~known_mask).unsqueeze(0).expand(n_use, -1)

            canonical_parts = []
            for start in range(0, n_use, batch_size):
                end = min(start + batch_size, n_use)
                z = teacher(x[start:end], total[start:end], mask_idx=None, unknown_mask=unknown_2d[start:end])
                canonical_parts.append(z.cpu())
            canonical = torch.cat(canonical_parts, dim=0)

            original_ge = teacher.gene_embeddings.data.clone()
            dataset_cos_sims = []

            for _ in range(n_permutations):
                perm = torch.randperm(num_genes, device=ctx.device)
                inv_perm = torch.argsort(perm)

                teacher.gene_embeddings.data = original_ge[perm]
                x_perm = x[:, perm]
                um_perm = unknown_2d[:, perm]

                perm_parts = []
                for start in range(0, n_use, batch_size):
                    end = min(start + batch_size, n_use)
                    z = teacher(x_perm[start:end], total[start:end], mask_idx=None, unknown_mask=um_perm[start:end])
                    perm_parts.append(z.cpu())
                perm_out = torch.cat(perm_parts, dim=0)

                teacher.gene_embeddings.data = original_ge

                z_unperm = perm_out[:, inv_perm.cpu()]
                cos = F.cosine_similarity(canonical[:, known_mask.cpu()], z_unperm[:, known_mask.cpu()], dim=-1)
                dataset_cos_sims.append(cos.numpy().flatten())

            teacher.gene_embeddings.data = original_ge

            flat = np.concatenate(dataset_cos_sims)
            by_dataset[ds_name] = {
                'mean_cosine_sim': float(np.mean(flat)),
                'std_cosine_sim': float(np.std(flat)),
                'min_cosine_sim': float(np.min(flat)),
                'n_known_genes': int(known_mask.sum().item()),
                'n_samples': n_use
            }
            all_cos_sims.append(flat)

            if verbose:
                print(f'permutation_invariance [{ds_name}]: mean={np.mean(flat):.6f}, min={np.min(flat):.6f}')

    random.setstate(rng_py)
    np.random.set_state(rng_np)
    torch.random.set_rng_state(rng_torch)
    if rng_cuda is not None:
        torch.cuda.set_rng_state(rng_cuda)

    all_flat = np.concatenate(all_cos_sims)
    result = {
        'config': {'n_permutations': n_permutations, 'max_samples_per_dataset': max_samples, 'n_datasets': len(by_dataset)},
        'overall': {'mean_cosine_sim': float(np.mean(all_flat)), 'std_cosine_sim': float(np.std(all_flat)), 'min_cosine_sim': float(np.min(all_flat))},
        'by_dataset': by_dataset
    }

    if verbose:
        print(f'permutation_invariance overall: mean={np.mean(all_flat):.6f}, min={np.min(all_flat):.6f}')

    return result


def summarize_encoder_evals(eval_results):
    '''Extract compact metrics dict from verbose eval results.'''
    summary = {}
    if 'perturbation_detection' in eval_results:
        summary['pert_auroc'] = eval_results['perturbation_detection'].get('metrics', {}).get('auroc')
    if 'latent_space_health' in eval_results:
        health = eval_results['latent_space_health']
        summary['eff_dim_90'] = health.get('effective_dimensionality', {}).get('90_percent')
        summary['dead_dims'] = health.get('variance', {}).get('n_dead_dims')
    if 'reconstruction' in eval_results:
        summary['recon_pearson'] = eval_results['reconstruction'].get('metrics', {}).get('pearson_r')
        summary['recon_r_squared'] = eval_results['reconstruction'].get('metrics', {}).get('pearson_r_squared')
    if 'batch_invariance' in eval_results:
        summary['invariance_ratio'] = eval_results['batch_invariance'].get('invariance_ratio')
    if 'gene_embedding_pathways' in eval_results:
        summary['kegg_similarity_ratio'] = eval_results['gene_embedding_pathways'].get('kegg', {}).get('similarity_ratio')
    if 'cell_type_probing' in eval_results:
        summary['cell_type_acc'] = eval_results['cell_type_probing'].get('metrics', {}).get('accuracy')
        summary['cell_type_macro_f1'] = eval_results['cell_type_probing'].get('metrics', {}).get('macro_f1')
    if 'essential_gene_prediction' in eval_results:
        summary['essential_auroc'] = eval_results['essential_gene_prediction'].get('classification', {}).get('auroc_test')
    if 'embedding_consistency' in eval_results:
        summary['consistency_ratio'] = eval_results['embedding_consistency'].get('metrics', {}).get('inter_intra_ratio')
    if 'permutation_invariance' in eval_results:
        summary['perm_invariance_cos'] = eval_results['permutation_invariance'].get('overall', {}).get('mean_cosine_sim')
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items() if v is not None}


# =============================================================================
# ENTRY POINTS
# =============================================================================

def run_encoder_evals(ctx):
    '''Run encoder training evaluations.'''
    return {
        'batch_invariance': _batch_invariance(ctx),
        'gene_embedding_pathways': _gene_embedding_pathways(ctx),
        'essential_gene_prediction': _essential_gene_prediction(ctx),
        'cell_type_probing': _cell_type_probing(ctx),
        'reconstruction': _reconstruction(ctx),
        'perturbation_detection': _perturbation_detection(ctx),
        'embedding_consistency': _embedding_consistency(ctx),
        'latent_space_health': _latent_space_health(ctx),
        'permutation_invariance': _permutation_invariance(ctx),
    }


def run_ac_evals(ctx):
    '''Run AC training evaluations.'''
    return {
        'expression_prediction': _expression_prediction(ctx),
        'gene_level_analysis': _gene_level_analysis(ctx),
        'perturbation_retrieval': _perturbation_retrieval(ctx),
        'uncertainty_calibration': _uncertainty_calibration(ctx),
        'moa_matching': _moa_matching(ctx),
        'combination_perturbation': _combination_perturbation(ctx),
        'dose_response': _dose_response(ctx),
    }


def run_composer_evals(ctx):
    '''Run composer training evaluations.'''
    return {
        'seq_to_target_retrieval': _seq_to_target_retrieval(ctx),
        'cross_modality_target_consistency': _cross_modality_target_consistency(ctx),
        'seq_target_gap_analysis': _seq_target_gap_analysis(ctx),
        'paired_alignment_quality': _paired_alignment_quality(ctx),
        'mode_sensitivity': _mode_sensitivity(ctx),
        'mode_semantic_consistency': _mode_semantic_consistency(ctx),
        'fusion_quality': _fusion_quality(ctx),
        'missing_data_robustness': _missing_data_robustness(ctx),
        'multi_pert_alignment': _multi_pert_alignment(ctx),
        'target_family_probing': _target_family_probing(ctx),
        'cross_modality_alignment': _cross_modality_alignment(ctx),
        'action_vector_pathways': _action_vector_pathways(ctx),
    }


# =============================================================================
# PRETRAINING EVALS
# =============================================================================

def _compute_batch_invariance(embeddings, batch_ids, pert_ids, device, seed):
    batch_map = {b: i for i, b in enumerate(sorted(np.unique(batch_ids)))}
    pert_map = {p: i for i, p in enumerate(sorted(np.unique(pert_ids)))}
    n_batch, n_pert = len(batch_map), len(pert_map)
    if n_batch < 2:
        return None
    batch_labels = np.array([batch_map[b] for b in batch_ids])
    pert_labels = np.array([pert_map[p] for p in pert_ids])
    train_idx, val_idx = train_test_split(np.arange(len(embeddings)), test_size=0.2, random_state=seed)
    _, _, batch_acc = train_linear_classifier(embeddings[train_idx], batch_labels[train_idx], embeddings[val_idx], batch_labels[val_idx], n_batch, device, epochs=100)
    _, _, pert_acc = train_linear_classifier(embeddings[train_idx], pert_labels[train_idx], embeddings[val_idx], pert_labels[val_idx], n_pert, device, epochs=100)
    batch_chance, pert_chance = 1.0 / n_batch, 1.0 / n_pert
    return {
        'config': {'samples': len(embeddings), 'embedding_dim': int(embeddings.shape[1]), 'num_batches': n_batch, 'num_perturbations': n_pert},
        'batch_classifier': {'accuracy': float(batch_acc), 'chance': float(batch_chance), 'above_chance_ratio': float(batch_acc / batch_chance)},
        'perturbation_classifier': {'accuracy': float(pert_acc), 'chance': float(pert_chance), 'above_chance_ratio': float(pert_acc / pert_chance)},
        'invariance_ratio': float(pert_acc / batch_acc) if batch_acc > 0 else 0.0
    }


def _batch_invariance(ctx):
    '''Are representations confounded by batch effects?'''
    verbose = ctx.config['verbose']
    eval_seed = ctx.config.get('seed', 1337)

    rng_py = random.getstate()
    rng_np = np.random.get_state()
    rng_torch = torch.random.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(eval_seed)

    test_loader = EvalLoader(batch_size=ctx.config['batch_size'], split=ctx.config.get('eval_split', 'test'), data_dir=ctx.paths['train_dir'], device=ctx.device, seed=ctx.config.get('seed', 1337))
    test_steps = ctx.config.get('test_total_examples', test_loader.total_samples) // ctx.config['batch_size']

    all_emb, all_batch, all_pert, all_ds_ids = [], [], [], []
    with torch.no_grad():
        for _ in tqdm(range(test_steps), desc='batch_invariance: Extracting embeddings', disable=not verbose):
            batch = test_loader.next_batch()
            cont_x, cont_tot = batch.control, batch.control_total
            unknown_mask = ~batch.gene_mask if hasattr(batch, 'gene_mask') and batch.gene_mask is not None else None
            all_emb.append(ctx.biojepa.teacher(cont_x, cont_tot, mask_idx=None, unknown_mask=unknown_mask).mean(dim=1).cpu().numpy())
            all_batch.append(batch.batch_id.cpu().numpy())
            all_pert.append(batch.seq_idx[:, 0].cpu().numpy())
            all_ds_ids.append(batch.dataset_id.cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0)
    batch_ids = np.concatenate(all_batch, axis=0).flatten()
    pert_ids = np.concatenate(all_pert, axis=0).flatten()
    dataset_ids = np.concatenate(all_ds_ids, axis=0).flatten()

    if verbose:
        print('Training classifiers...')
    result = _compute_batch_invariance(embeddings, batch_ids, pert_ids, ctx.device, eval_seed)

    random.setstate(rng_py)
    np.random.set_state(rng_np)
    torch.random.set_rng_state(rng_torch)
    if rng_cuda is not None:
        torch.cuda.set_rng_state(rng_cuda)

    if verbose:
        ba = result['batch_classifier']['accuracy']
        pa = result['perturbation_classifier']['accuracy']
        print(f'batch_invariance: Batch={ba:.4f} ({ba/result["batch_classifier"]["chance"]:.1f}x), Pert={pa:.4f} ({pa/result["perturbation_classifier"]["chance"]:.1f}x)')

    by_dataset = {}
    excluded_datasets = {}
    for ds_id, ds_name in test_loader.dataset_id_to_name.items():
        mask = dataset_ids == ds_id
        sample_count = int(mask.sum())
        if sample_count < 200:
            excluded_datasets[ds_name] = {'samples': sample_count, 'reason': 'fewer than 200 samples'}
            continue
        ds_result = _compute_batch_invariance(embeddings[mask], batch_ids[mask], pert_ids[mask], ctx.device, eval_seed)
        if ds_result is not None:
            by_dataset[ds_name] = ds_result
    result['by_dataset'] = by_dataset
    result['scope'] = 'cross_dataset_aggregate'
    result['interpretation_note'] = (
        'Global metrics are aggregated across datasets and may include dataset-composition effects. '
        'Use by_dataset and within_dataset_summary for cleaner within-dataset interpretation.'
    )
    result['excluded_datasets'] = excluded_datasets

    if by_dataset:
        ratios = np.array([v['invariance_ratio'] for v in by_dataset.values()], dtype=np.float64)
        batch_accs = np.array([v['batch_classifier']['accuracy'] for v in by_dataset.values()], dtype=np.float64)
        pert_accs = np.array([v['perturbation_classifier']['accuracy'] for v in by_dataset.values()], dtype=np.float64)
        weights = np.array([v['config']['samples'] for v in by_dataset.values()], dtype=np.float64)
        weight_sum = float(weights.sum())
        result['within_dataset_summary'] = {
            'n_datasets_included': int(len(by_dataset)),
            'macro_mean': {
                'invariance_ratio': float(ratios.mean()),
                'batch_accuracy': float(batch_accs.mean()),
                'perturbation_accuracy': float(pert_accs.mean()),
            },
            'weighted_mean': {
                'invariance_ratio': float(np.dot(ratios, weights) / weight_sum) if weight_sum > 0 else 0.0,
                'batch_accuracy': float(np.dot(batch_accs, weights) / weight_sum) if weight_sum > 0 else 0.0,
                'perturbation_accuracy': float(np.dot(pert_accs, weights) / weight_sum) if weight_sum > 0 else 0.0,
            },
        }
        if verbose:
            g_ratio = result['invariance_ratio']
            m_ratio = result['within_dataset_summary']['macro_mean']['invariance_ratio']
            print(f'batch_invariance summary: global_ratio={g_ratio:.3f}, within_dataset_macro_ratio={m_ratio:.3f}')
    else:
        result['within_dataset_summary'] = {
            'n_datasets_included': 0,
            'macro_mean': {'invariance_ratio': 0.0, 'batch_accuracy': 0.0, 'perturbation_accuracy': 0.0},
            'weighted_mean': {'invariance_ratio': 0.0, 'batch_accuracy': 0.0, 'perturbation_accuracy': 0.0},
        }
    return result


def _gene_embedding_pathways(ctx):
    '''Do genes in same pathway cluster in learned embeddings?'''
    verbose = ctx.config['verbose']
    pathway_libs = ctx.pathway_annotations
    gene_names_upper = [g.upper() for g in ctx.gene_names]
    results = {'config': {'n_genes': len(ctx.gene_names)}}

    for lib_key, lib_name in [('kegg', 'KEGG_2026'), ('reactome', 'Reactome_Pathways_2024')]:
        gene_to_pathways = build_gene_to_pathways(pathway_libs[lib_name], min_pathway_size=15, max_pathway_size=300)
        idx_list, labels_list = [], []
        for i, gene in enumerate(gene_names_upper):
            if gene in gene_to_pathways:
                idx_list.append(i)
                labels_list.append(gene_to_pathways[gene])
        metrics = compute_multilabel_pathway_similarity(ctx.gene_embeddings[idx_list], labels_list, min_pathway_size=10)
        results[lib_key] = metrics

    if verbose and 'error' not in results.get('kegg', {}):
        print(f'gene_embedding_pathways: KEGG ratio={results["kegg"]["similarity_ratio"]:.4f}')
    return results


def _essential_gene_prediction(ctx):
    '''Do gene embeddings encode functional importance?'''
    verbose = ctx.config['verbose']
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

    r, _ = pearsonr(y_test, y_pred_test)
    pearson_test = 0.0 if np.isnan(r) else float(r)
    r, _ = spearmanr(y_test, y_pred_test)
    spearman_test = 0.0 if np.isnan(r) else float(r)

    THRESH = -0.5
    y_test_bin = (y_test < THRESH).astype(int)
    auroc_test = roc_auc_score(y_test_bin, -y_pred_test)

    if verbose:
        print(f'essential_gene_prediction: Pearson={pearson_test:.4f}, AUROC={auroc_test:.4f}')

    return {
        'config': {'matched_genes': len(matched_idx), 'train_genes': len(X_train), 'test_genes': len(X_test)},
        'regression': {'pearson_test': float(pearson_test), 'spearman_test': float(spearman_test)},
        'classification': {'auroc_test': float(auroc_test), 'n_essential_test': int(y_test_bin.sum()), 'n_non_essential_test': int((~y_test_bin.astype(bool)).sum())}
    }


def _cell_type_probing(ctx):
    '''Can cell type be predicted from cell embeddings?'''
    verbose = ctx.config['verbose']
    test_loader = TrainingLoader(batch_size=ctx.config['batch_size'], split=ctx.config.get('eval_split', 'test'), data_dir=ctx.paths['train_dir'], device=ctx.device, seed=ctx.config.get('seed', 1337))
    test_steps = ctx.config.get('test_total_examples', test_loader.total_samples) // ctx.config['batch_size']

    all_emb, all_cell_type, all_batch_id = [], [], []
    with torch.no_grad():
        for _ in tqdm(range(test_steps), desc='cell_type_probing: Extracting embeddings', disable=not verbose):
            batch = test_loader.next_batch()
            cont_x, cont_tot = batch.control, batch.control_total
            unknown_mask = ~batch.gene_mask if hasattr(batch, 'gene_mask') and batch.gene_mask is not None else None
            emb = ctx.biojepa.teacher(cont_x, cont_tot, mask_idx=None, unknown_mask=unknown_mask).mean(dim=1).cpu().numpy()
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

    if verbose:
        print('Training cell type classifier...')
    _, val_preds, val_acc = train_linear_classifier(embeddings[train_idx], labels[train_idx], embeddings[val_idx], labels[val_idx], n_classes, ctx.device, epochs=100)

    macro_f1 = f1_score(labels[val_idx], val_preds, average='macro')
    chance = 1.0 / n_classes

    if verbose:
        print(f'cell_type_probing: Accuracy={val_acc:.4f} ({val_acc/chance:.1f}x chance), Macro F1={macro_f1:.4f}')

    return {
        'config': {'samples': len(embeddings), 'embedding_dim': int(embeddings.shape[1]), 'num_cell_types': n_classes, 'filtered_from': len(unique_types)},
        'metrics': {'accuracy': float(val_acc), 'macro_f1': float(macro_f1), 'chance': float(chance), 'above_chance_ratio': float(val_acc / chance)}
    }


def _reconstruction(ctx):
    '''Can gene expression be reconstructed from embeddings?'''
    verbose = ctx.config['verbose']
    test_loader = TrainingLoader(batch_size=ctx.config['batch_size'], split=ctx.config.get('eval_split', 'test'), data_dir=ctx.paths['train_dir'], device=ctx.device, seed=ctx.config.get('seed', 1337))
    recon_samples = min(ctx.config.get('test_total_examples', test_loader.total_samples), 100)
    test_steps = max(recon_samples // ctx.config['batch_size'], 1)
    n_genes = ctx.config['num_genes']

    all_emb, all_expr = [], []
    all_gene_masks = []
    with torch.no_grad():
        for _ in tqdm(range(test_steps), desc='reconstruction: Extracting embeddings', disable=not verbose):
            batch = test_loader.next_batch()
            cont_x, cont_tot = batch.control, batch.control_total
            unknown_mask = ~batch.gene_mask if hasattr(batch, 'gene_mask') and batch.gene_mask is not None else None
            emb = ctx.biojepa.teacher(cont_x, cont_tot, mask_idx=None, unknown_mask=unknown_mask).cpu().numpy()
            all_emb.append(emb)
            all_expr.append(cont_x.cpu().numpy())
            if hasattr(batch, 'gene_mask') and batch.gene_mask is not None:
                all_gene_masks.append(batch.gene_mask.cpu().numpy().astype(np.bool_))

    embeddings = np.concatenate(all_emb, axis=0)
    expressions = np.concatenate(all_expr, axis=0)
    gene_masks = np.concatenate(all_gene_masks, axis=0) if all_gene_masks else None

    n_samples = embeddings.shape[0]
    gene_perm = np.random.RandomState(42).permutation(n_genes)
    n_train_genes = int(0.8 * n_genes)
    train_genes, test_genes = gene_perm[:n_train_genes], gene_perm[n_train_genes:]

    X_train = embeddings[:, train_genes, :].reshape(-1, embeddings.shape[-1])
    y_train = expressions[:, train_genes].reshape(-1)
    X_test = embeddings[:, test_genes, :].reshape(-1, embeddings.shape[-1])
    y_test = expressions[:, test_genes].reshape(-1)
    if gene_masks is not None:
        train_measured = gene_masks[:, train_genes].reshape(-1)
        test_measured = gene_masks[:, test_genes].reshape(-1)
        X_train, y_train = X_train[train_measured], y_train[train_measured]
        X_test, y_test = X_test[test_measured], y_test[test_measured]

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

    if verbose:
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
    r, _ = pearsonr(y_pred, y_true)
    pearson_r = 0.0 if np.isnan(r) else float(r)

    if verbose:
        print(f'reconstruction: MSE={mse:.4f}, Pearson R={pearson_r:.4f}')

    return {
        'config': {'samples': n_samples, 'train_genes': len(train_genes), 'test_genes': len(test_genes), 'embedding_dim': int(embeddings.shape[-1])},
        'metrics': {'reconstruction_mse': float(mse), 'pearson_r': pearson_r, 'pearson_r_squared': pearson_r**2}
    }


def _compute_perturbation_detection(control_emb, case_emb, device):
    X = np.concatenate([control_emb, case_emb], axis=0)
    y = np.concatenate([np.zeros(len(control_emb)), np.ones(len(case_emb))]).astype(int)
    train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)
    classifier, val_preds, val_acc = train_linear_classifier(X[train_idx], y[train_idx], X[val_idx], y[val_idx], num_classes=2, device=device, epochs=100)
    classifier.eval()
    with torch.no_grad():
        X_val_t = torch.from_numpy(X[val_idx]).float().to(device)
        logits = classifier(X_val_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    auroc = roc_auc_score(y[val_idx], probs)
    return {
        'config': {'n_control': len(control_emb), 'n_perturbed': len(case_emb), 'embedding_dim': int(control_emb.shape[1])},
        'metrics': {'auroc': float(auroc), 'accuracy': float(val_acc), 'chance': 0.5}
    }


def _perturbation_detection(ctx):
    '''Can we distinguish perturbed cells from control cells?'''
    verbose = ctx.config['verbose']
    test_loader = EvalLoader(batch_size=ctx.config['batch_size'], split=ctx.config.get('eval_split', 'test'), data_dir=ctx.paths['train_dir'], device=ctx.device, seed=ctx.config.get('seed', 1337))
    test_steps = ctx.config.get('test_total_examples', test_loader.total_samples) // ctx.config['batch_size']

    control_emb, case_emb, all_ds_ids = [], [], []
    with torch.no_grad():
        for _ in tqdm(range(test_steps), desc='perturbation_detection: Extracting embeddings', disable=not verbose):
            batch = test_loader.next_batch()
            cont_x, cont_tot = batch.control, batch.control_total
            case_x, case_tot = batch.case, batch.case_total
            unknown_mask = ~batch.gene_mask if hasattr(batch, 'gene_mask') and batch.gene_mask is not None else None
            ctrl_z = ctx.biojepa.teacher(cont_x, cont_tot, mask_idx=None, unknown_mask=unknown_mask).mean(dim=1).cpu().numpy()
            case_z = ctx.biojepa.teacher(case_x, case_tot, mask_idx=None, unknown_mask=unknown_mask).mean(dim=1).cpu().numpy()
            control_emb.append(ctrl_z)
            case_emb.append(case_z)
            all_ds_ids.append(batch.dataset_id.cpu().numpy())

    control_emb = np.concatenate(control_emb, axis=0)
    case_emb = np.concatenate(case_emb, axis=0)
    dataset_ids = np.concatenate(all_ds_ids, axis=0).flatten()

    if verbose:
        print('Training perturbation detector...')
    result = _compute_perturbation_detection(control_emb, case_emb, ctx.device)
    if verbose:
        print(f'perturbation_detection: AUROC={result["metrics"]["auroc"]:.4f}, Accuracy={result["metrics"]["accuracy"]:.4f}')

    by_dataset = {}
    for ds_id, ds_name in test_loader.dataset_id_to_name.items():
        mask = dataset_ids == ds_id
        if mask.sum() < 200:
            continue
        by_dataset[ds_name] = _compute_perturbation_detection(control_emb[mask], case_emb[mask], ctx.device)
    result['by_dataset'] = by_dataset
    return result


def _compute_embedding_consistency(embeddings, pert_ids, label='', verbose=False):
    pert_to_emb = defaultdict(list)
    for i, pid in enumerate(pert_ids):
        pert_to_emb[pid].append(embeddings[i])
    valid_perts = {pid: np.array(embs) for pid, embs in pert_to_emb.items() if len(embs) >= 3}
    if len(valid_perts) < 10:
        return {'error': f'Not enough perturbations with >= 3 replicates (found {len(valid_perts)})'}

    prefix = f'embedding_consistency{label}'
    if verbose:
        print(f'{prefix}: Computing intra-distances for {len(valid_perts)} perturbations...')
    intra_dists = np.concatenate([pdist(embs) for embs in valid_perts.values()])

    pert_list = list(valid_perts.keys())
    n_samples = min(5000, len(pert_list) * (len(pert_list) - 1) // 2)
    if verbose:
        print(f'{prefix}: Computing {n_samples} inter-distances...')
    rng = np.random.RandomState(42)
    e1_list, e2_list = [], []
    for _ in range(n_samples):
        idx1, idx2 = rng.choice(len(pert_list), 2, replace=False)
        p1, p2 = pert_list[idx1], pert_list[idx2]
        e1_list.append(valid_perts[p1][rng.randint(len(valid_perts[p1]))])
        e2_list.append(valid_perts[p2][rng.randint(len(valid_perts[p2]))])
    inter_dists = np.linalg.norm(np.array(e1_list) - np.array(e2_list), axis=1)

    intra_mean, inter_mean = np.mean(intra_dists), np.mean(inter_dists)
    ratio = inter_mean / intra_mean if intra_mean > 0 else float('inf')
    return {
        'config': {'n_perturbations': len(valid_perts), 'n_intra_pairs': len(intra_dists), 'n_inter_pairs': len(inter_dists)},
        'metrics': {'mean_intra_distance': float(intra_mean), 'mean_inter_distance': float(inter_mean), 'inter_intra_ratio': float(ratio), 'std_intra_distance': float(np.std(intra_dists)), 'std_inter_distance': float(np.std(inter_dists))}
    }


def _embedding_consistency(ctx):
    '''Do replicates of the same perturbation cluster together?'''
    verbose = ctx.config['verbose']
    test_loader = EvalLoader(batch_size=ctx.config['batch_size'], split=ctx.config.get('eval_split', 'test'), data_dir=ctx.paths['train_dir'], device=ctx.device, seed=ctx.config.get('seed', 1337))
    test_steps = ctx.config.get('test_total_examples', test_loader.total_samples) // ctx.config['batch_size']

    all_emb, all_seq, all_target, all_mod, all_mode, all_ct, all_ds_ids = [], [], [], [], [], [], []
    with torch.no_grad():
        for _ in tqdm(range(test_steps), desc='embedding_consistency: Extracting embeddings', disable=not verbose):
            batch = test_loader.next_batch()
            case_x, case_tot = batch.case, batch.case_total
            unknown_mask = ~batch.gene_mask if hasattr(batch, 'gene_mask') and batch.gene_mask is not None else None
            case_z = ctx.biojepa.teacher(case_x, case_tot, mask_idx=None, unknown_mask=unknown_mask).mean(dim=1).cpu().numpy()
            all_emb.append(case_z)
            all_seq.append(batch.seq_idx[:, 0].cpu().numpy())
            all_target.append(batch.target_idx[:, 0].cpu().numpy())
            all_mod.append(batch.modality[:, 0].cpu().numpy())
            all_mode.append(batch.mode[:, 0].cpu().numpy())
            all_ct.append(batch.cell_type.cpu().numpy())
            all_ds_ids.append(batch.dataset_id.cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0)
    seq_ids = np.concatenate(all_seq, axis=0)
    target_ids = np.concatenate(all_target, axis=0)
    mod_ids = np.concatenate(all_mod, axis=0)
    mode_ids = np.concatenate(all_mode, axis=0)
    ct_ids = np.concatenate(all_ct, axis=0)
    dataset_ids = np.concatenate(all_ds_ids, axis=0).flatten()
    pert_ids = [(int(seq_ids[i]), int(target_ids[i]), int(mod_ids[i]), int(mode_ids[i]), int(ct_ids[i])) for i in range(len(seq_ids))]

    result = _compute_embedding_consistency(embeddings, pert_ids, verbose=verbose)
    if verbose:
        if 'error' not in result:
            print(f'embedding_consistency: Intra={result["metrics"]["mean_intra_distance"]:.4f}, Inter={result["metrics"]["mean_inter_distance"]:.4f}, Ratio={result["metrics"]["inter_intra_ratio"]:.2f}x')
        else:
            print(f'embedding_consistency: {result["error"]}')

    by_dataset = {}
    for ds_id, ds_name in test_loader.dataset_id_to_name.items():
        mask = dataset_ids == ds_id
        if mask.sum() < 200:
            continue
        if verbose:
            print(f'embedding_consistency: Computing for dataset {ds_name}...')
        ds_result = _compute_embedding_consistency(embeddings[mask], [pert_ids[i] for i in np.where(mask)[0]])
        if 'error' not in ds_result:
            by_dataset[ds_name] = ds_result
    result['by_dataset'] = by_dataset
    return result


def _latent_space_health(ctx):
    '''Diagnostic metrics for embedding quality.'''
    verbose = ctx.config['verbose']
    test_loader = TrainingLoader(batch_size=ctx.config['batch_size'], split=ctx.config.get('eval_split', 'test'), data_dir=ctx.paths['train_dir'], device=ctx.device, seed=ctx.config.get('seed', 1337))
    test_steps = ctx.config.get('test_total_examples', test_loader.total_samples) // ctx.config['batch_size']

    all_emb = []
    with torch.no_grad():
        for _ in tqdm(range(test_steps), desc='latent_space_health: Extracting embeddings', disable=not verbose):
            batch = test_loader.next_batch()
            cont_x, cont_tot = batch.control, batch.control_total
            unknown_mask = ~batch.gene_mask if hasattr(batch, 'gene_mask') and batch.gene_mask is not None else None
            emb = ctx.biojepa.teacher(cont_x, cont_tot, mask_idx=None, unknown_mask=unknown_mask).mean(dim=1).cpu().numpy()
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

    if verbose:
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

def _compute_expression_prediction(inf, n_genes, pert_gene_masks=None):
    pert_keys = inf['pert_keys']
    mean_pred_deltas, mean_real_deltas = inf['mean_pred_deltas'], inf['mean_real_deltas']
    mean_pred_abs, mean_real_abs = inf['mean_pred_abs'], inf['mean_real_abs']
    mean_control_states = inf['mean_control_states']
    sample_mses, sample_correlations = inf['sample_mses'], inf['sample_correlations']

    def _masked(arr, key):
        if pert_gene_masks and key in pert_gene_masks:
            return arr[pert_gene_masks[key]]
        return arr

    TOP_K = 50
    per_pert_r2_all, per_pert_r2_top50, per_pert_mse = [], [], []
    per_pert_pearson_abs, per_pert_pearson_delta, per_pert_pearson_top50 = [], [], []
    for key in pert_keys:
        pred_abs, real_abs = _masked(mean_pred_abs[key], key), _masked(mean_real_abs[key], key)
        pred_delta, real_delta = _masked(mean_pred_deltas[key], key), _masked(mean_real_deltas[key], key)
        if np.std(real_abs) > 1e-9:
            per_pert_r2_all.append(r2_score(real_abs, pred_abs))
        top_k_idx = np.argsort(np.abs(real_delta))[-TOP_K:]
        per_pert_r2_top50.append(r2_score(real_abs[top_k_idx], pred_abs[top_k_idx]))
        per_pert_mse.append(np.mean((pred_delta - real_delta)**2))

        if np.std(pred_abs) > 1e-9 and np.std(real_abs) > 1e-9:
            r, _ = pearsonr(pred_abs, real_abs)
            per_pert_pearson_abs.append(0.0 if np.isnan(r) else float(r))
        else:
            per_pert_pearson_abs.append(0.0)

        if np.std(pred_delta) > 1e-9 and np.std(real_delta) > 1e-9:
            r, _ = pearsonr(pred_delta, real_delta)
            per_pert_pearson_delta.append(0.0 if np.isnan(r) else float(r))
        else:
            per_pert_pearson_delta.append(0.0)

        pd_top, rd_top = pred_delta[top_k_idx], real_delta[top_k_idx]
        if np.std(pd_top) > 1e-9 and np.std(rd_top) > 1e-9:
            r, _ = pearsonr(pd_top, rd_top)
            per_pert_pearson_top50.append(0.0 if np.isnan(r) else float(r))
        else:
            per_pert_pearson_top50.append(0.0)

    per_pert_r2_all, per_pert_r2_top50 = np.array(per_pert_r2_all), np.array(per_pert_r2_top50)
    per_pert_mse = np.array(per_pert_mse)
    per_pert_pearson_abs = np.array(per_pert_pearson_abs)
    per_pert_pearson_delta = np.array(per_pert_pearson_delta)
    per_pert_pearson_top50 = np.array(per_pert_pearson_top50)

    if len(pert_keys) >= 2 and len(pert_keys[0]) == 5:
        group_to_keys = defaultdict(list)
        for key in pert_keys:
            seq_idx, target_idx, modality, mode, cell_type = key
            if target_idx >= 0:
                group = ('t', target_idx, mode, cell_type)
            else:
                group = ('s', seq_idx, mode, cell_type)
            group_to_keys[group].append(key)

        groups = list(group_to_keys.keys())
        if pert_gene_masks:
            common_mask = np.ones(len(mean_pred_deltas[pert_keys[0]]), dtype=bool)
            for key in pert_keys:
                km = pert_gene_masks.get(key)
                if km is not None:
                    common_mask &= km
            grouped_pred = np.array([np.mean([mean_pred_deltas[k][common_mask] for k in group_to_keys[g]], axis=0) for g in groups])
            grouped_real = np.array([np.mean([mean_real_deltas[k][common_mask] for k in group_to_keys[g]], axis=0) for g in groups])
        else:
            grouped_pred = np.array([np.mean([mean_pred_deltas[k] for k in group_to_keys[g]], axis=0) for g in groups])
            grouped_real = np.array([np.mean([mean_real_deltas[k] for k in group_to_keys[g]], axis=0) for g in groups])
        g_pred_sq = np.sum(grouped_pred**2, axis=1)
        g_real_sq = np.sum(grouped_real**2, axis=1)
        g_dist_matrix = g_pred_sq[:, None] + g_real_sq[None, :] - 2.0 * grouped_pred @ grouped_real.T
        centroid_acc = float(np.mean(np.argmin(g_dist_matrix, axis=1) == np.arange(len(groups))))
    else:
        centroid_acc = 0.0
        groups = []

    n_beat, n_eval_baseline = 0, 0
    for key in pert_keys:
        real_abs = _masked(mean_real_abs[key], key)
        if np.std(real_abs) < 1e-9:
            continue
        pred_abs = _masked(mean_pred_abs[key], key)
        control = _masked(mean_control_states[key], key)
        if np.std(pred_abs) > 1e-9:
            r_model, _ = pearsonr(pred_abs, real_abs)
        else:
            r_model = 0.0
        if np.std(control) > 1e-9:
            r_baseline, _ = pearsonr(control, real_abs)
        else:
            r_baseline = 0.0
        r_model = 0.0 if np.isnan(r_model) else r_model
        r_baseline = 0.0 if np.isnan(r_baseline) else r_baseline
        n_eval_baseline += 1
        if r_model > r_baseline:
            n_beat += 1

    pred_severity = np.array([np.linalg.norm(_masked(mean_pred_deltas[k], k)) for k in pert_keys])
    real_severity = np.array([np.linalg.norm(_masked(mean_real_deltas[k], k)) for k in pert_keys])
    r, _ = pearsonr(pred_severity, real_severity)
    severity_pearson = 0.0 if np.isnan(r) else float(r)
    r, _ = spearmanr(pred_severity, real_severity)
    severity_spearman = 0.0 if np.isnan(r) else float(r)

    all_pred = np.concatenate([_masked(mean_pred_deltas[k], k) for k in pert_keys])
    all_real = np.concatenate([_masked(mean_real_deltas[k], k) for k in pert_keys])
    all_errors, all_magnitudes = all_pred - all_real, np.abs(all_real)
    magnitude_bins = [0, 0.25, 0.5, 1.0, 1.5, 2.0, np.inf]
    bin_labels = ['0-0.25', '0.25-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0+']
    error_by_magnitude = {}
    for i in range(len(magnitude_bins) - 1):
        mask = (all_magnitudes >= magnitude_bins[i]) & (all_magnitudes < magnitude_bins[i + 1])
        if mask.sum() > 0:
            error_by_magnitude[bin_labels[i]] = {'mae': float(np.mean(np.abs(all_errors[mask]))), 'count': int(mask.sum())}

    return {
        'config': {'test_perturbations': len(pert_keys), 'genes': n_genes, 'test_samples': len(sample_mses)},
        'sample_level': {'mse': float(np.mean(sample_mses)), 'pearson_r_top20': float(np.mean(sample_correlations))},
        'perturbation_level': {
            'r2_all_genes': {'mean': float(per_pert_r2_all.mean()), 'median': float(np.median(per_pert_r2_all))},
            'r2_top50_degs': {'mean': float(per_pert_r2_top50.mean()), 'median': float(np.median(per_pert_r2_top50))},
            'mse': {'mean': float(per_pert_mse.mean()), 'median': float(np.median(per_pert_mse))},
            'pearson_all_genes': {'mean': float(per_pert_pearson_abs.mean()), 'median': float(np.median(per_pert_pearson_abs))},
            'pearson_delta_all_genes': {'mean': float(per_pert_pearson_delta.mean()), 'median': float(np.median(per_pert_pearson_delta))},
            'pearson_top50_degs': {'mean': float(per_pert_pearson_top50.mean()), 'median': float(np.median(per_pert_pearson_top50))},
        },
        'centroid_accuracy': {'accuracy': centroid_acc, 'n_groups': len(groups)},
        'vs_baseline': {'beat_rate': float(n_beat / n_eval_baseline) if n_eval_baseline > 0 else 0.0, 'n_evaluated': n_eval_baseline},
        'severity': {'pearson_r': float(severity_pearson), 'spearman_r': float(severity_spearman)},
        'error_by_magnitude': error_by_magnitude
    }


def _gears_benchmark_summary(result):
    summary = {}
    by_ds = result.get('by_dataset', {})
    for ds_name, gears_name in [('k562e_raw', 'replogle_k562'), ('adamson', 'adamson')]:
        ds = by_ds.get(ds_name)
        if not ds:
            continue
        pl = ds.get('perturbation_level', {})
        summary[gears_name] = {
            'pearson_all_genes': pl.get('pearson_all_genes', {}),
            'pearson_delta_all_genes': pl.get('pearson_delta_all_genes', {}),
            'r2_all_genes': pl.get('r2_all_genes', {}),
            'centroid_accuracy': ds.get('centroid_accuracy'),
            'n_test_perturbations': ds.get('config', {}).get('test_perturbations'),
            'uses_gears_official_splits': ds_name == 'k562e_raw',
        }
    return summary


def _expression_prediction(ctx):
    '''Can we predict gene expression after perturbation?'''
    inf = ctx.test_inference
    n_genes = ctx.config['num_genes']
    result = _compute_expression_prediction(inf, n_genes, pert_gene_masks=inf.get('pert_gene_masks'))
    print(f'expression_prediction: Pearson={result["perturbation_level"]["pearson_all_genes"]["mean"]:.4f}, R2={result["perturbation_level"]["r2_all_genes"]["mean"]:.4f}, Centroid_acc={result["centroid_accuracy"]["accuracy"]:.4f}')
    by_dataset = {}
    for ds, ds_inf in inf.get('by_dataset', {}).items():
        if len(ds_inf.get('pert_keys', [])) > 0:
            by_dataset[ds] = _compute_expression_prediction(ds_inf, n_genes, pert_gene_masks=ds_inf.get('pert_gene_masks'))
    result['by_dataset'] = by_dataset
    result['gears_benchmark'] = _gears_benchmark_summary(result)

    by_cell_type = inf.get('by_cell_type', {})
    if len(by_cell_type) > 1:
        ct_id_to_name = {}
        ct_map_path = ctx.paths['data_dir'] / 'cell_type_to_id.json'
        if ct_map_path.exists():
            with open(ct_map_path) as f:
                ct_id_to_name = {v: k for k, v in json.load(f).items()}
        result['by_cell_type'] = {}
        for ct_id, ct_inf in by_cell_type.items():
            if len(ct_inf.get('pert_keys', [])) > 0:
                ct_name = ct_id_to_name.get(ct_id, f'cell_type_{ct_id}')
                result['by_cell_type'][ct_name] = _compute_expression_prediction(ct_inf, n_genes, pert_gene_masks=ct_inf.get('pert_gene_masks'))

    return result


def _classify_direction(delta, threshold):
    direction = np.zeros_like(delta, dtype=np.int8)
    direction[delta >= threshold] = 1
    direction[delta <= -threshold] = -1
    return direction


def _precision_at_k(pred_rank, true_rank, k):
    return len(set(pred_rank[:k]) & set(true_rank[:k])) / k


def _ndcg_at_k(pred_rank, true_rank, k):
    true_set = set(true_rank[:k])
    rels = [1 if g in true_set else 0 for g in pred_rank[:k]]
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rels))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(true_set))))
    return dcg / idcg if idcg > 0 else 0.0


def _compute_gene_level_analysis(inf, n_genes, direction_threshold, pert_gene_masks=None):
    pert_keys = inf['pert_keys']
    mean_pred_deltas, mean_real_deltas = inf['mean_pred_deltas'], inf['mean_real_deltas']

    def _masked(arr, key):
        if pert_gene_masks and key in pert_gene_masks:
            return arr[pert_gene_masks[key]]
        return arr

    all_pred_dir = np.concatenate([_classify_direction(_masked(mean_pred_deltas[k], k), direction_threshold) for k in pert_keys])
    all_real_dir = np.concatenate([_classify_direction(_masked(mean_real_deltas[k], k), direction_threshold) for k in pert_keys])
    overall_accuracy = accuracy_score(all_real_dir, all_pred_dir)
    f1_up = f1_score(all_real_dir, all_pred_dir, labels=[1], average='macro', zero_division=0)
    f1_down = f1_score(all_real_dir, all_pred_dir, labels=[-1], average='macro', zero_division=0)
    f1_unchanged = f1_score(all_real_dir, all_pred_dir, labels=[0], average='macro', zero_division=0)

    TOP_K_DIR = 50
    top_deg_pred, top_deg_real = [], []
    for key in pert_keys:
        pd, rd = _masked(mean_pred_deltas[key], key), _masked(mean_real_deltas[key], key)
        top_k_idx = np.argsort(np.abs(rd))[-TOP_K_DIR:]
        top_deg_pred.append(_classify_direction(pd[top_k_idx], direction_threshold))
        top_deg_real.append(_classify_direction(rd[top_k_idx], direction_threshold))
    top_deg_accuracy = accuracy_score(np.concatenate(top_deg_real), np.concatenate(top_deg_pred))

    all_magnitudes, all_correct = [], []
    for key in pert_keys:
        real_delta, pred_delta = _masked(mean_real_deltas[key], key), _masked(mean_pred_deltas[key], key)
        all_magnitudes.extend(np.abs(real_delta))
        all_correct.extend(_classify_direction(pred_delta, direction_threshold) == _classify_direction(real_delta, direction_threshold))
    all_magnitudes, all_correct = np.array(all_magnitudes), np.array(all_correct)

    magnitude_bins = [0, 0.25, 0.5, 1.0, 1.5, 2.0, np.inf]
    bin_labels = ['0-0.25', '0.25-0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0+']
    accuracy_by_magnitude = {}
    for i in range(len(magnitude_bins) - 1):
        mask = (all_magnitudes >= magnitude_bins[i]) & (all_magnitudes < magnitude_bins[i + 1])
        if mask.sum() > 0:
            accuracy_by_magnitude[bin_labels[i]] = {'accuracy': float(all_correct[mask].mean()), 'count': int(mask.sum())}

    K_VALUES = [10, 20, 50, 100]
    deg_results = {k: {'precision': [], 'ndcg': [], 'overlap': [], 'vs_random': []} for k in K_VALUES}
    for key in pert_keys:
        pd, rd = _masked(mean_pred_deltas[key], key), _masked(mean_real_deltas[key], key)
        n_measured = len(pd)
        pred_rank = np.argsort(np.abs(pd))[::-1]
        true_rank = np.argsort(np.abs(rd))[::-1]
        for k in K_VALUES:
            deg_results[k]['precision'].append(_precision_at_k(pred_rank, true_rank, k))
            deg_results[k]['ndcg'].append(_ndcg_at_k(pred_rank, true_rank, k))
            overlap = len(set(pred_rank[:k]) & set(true_rank[:k]))
            deg_results[k]['overlap'].append(overlap)
            deg_results[k]['vs_random'].append(overlap / (k * k / n_measured) if n_measured > 0 else 0.0)

    K_DIR = 20
    up_precisions, down_precisions = [], []
    de_jaccards = []
    for key in pert_keys:
        pred_d, real_d = _masked(mean_pred_deltas[key], key), _masked(mean_real_deltas[key], key)
        pred_up, real_up = set(np.argsort(pred_d)[-K_DIR:]), set(np.argsort(real_d)[-K_DIR:])
        pred_down, real_down = set(np.argsort(pred_d)[:K_DIR]), set(np.argsort(real_d)[:K_DIR])
        up_precisions.append(len(pred_up & real_up) / K_DIR)
        down_precisions.append(len(pred_down & real_down) / K_DIR)
        pred_de = set(np.where(np.abs(pred_d) > direction_threshold)[0])
        real_de = set(np.where(np.abs(real_d) > direction_threshold)[0])
        union = len(pred_de | real_de)
        if union > 0:
            de_jaccards.append(len(pred_de & real_de) / union)

    result = {
        'config': {'test_perturbations': len(pert_keys), 'genes': n_genes, 'direction_threshold': direction_threshold},
        'direction_of_effect': {
            'all_genes_accuracy': float(overall_accuracy), 'top50_degs_accuracy': float(top_deg_accuracy),
            'f1_up': float(f1_up), 'f1_down': float(f1_down), 'f1_unchanged': float(f1_unchanged),
            'accuracy_by_magnitude': accuracy_by_magnitude,
            'precision_up_at_20': float(np.mean(up_precisions)), 'precision_down_at_20': float(np.mean(down_precisions)),
        },
        'top_deg_recovery': {str(k): {'precision': float(np.mean(deg_results[k]['precision'])), 'ndcg': float(np.mean(deg_results[k]['ndcg'])), 'overlap': float(np.mean(deg_results[k]['overlap'])), 'vs_random': float(np.mean(deg_results[k]['vs_random']))} for k in K_VALUES},
    }
    if de_jaccards:
        result['common_degs'] = {'jaccard_mean': float(np.mean(de_jaccards)), 'jaccard_median': float(np.median(de_jaccards)), 'n_perturbations': len(de_jaccards)}
    return result


def _gene_level_analysis(ctx, direction_threshold=0.25):
    '''Direction of effect + top DEG recovery analysis.'''
    inf = ctx.test_inference
    n_genes = ctx.config['num_genes']
    result = _compute_gene_level_analysis(inf, n_genes, direction_threshold, pert_gene_masks=inf.get('pert_gene_masks'))
    print(f'gene_level_analysis: Dir_acc={result["direction_of_effect"]["all_genes_accuracy"]:.4f}, Top50_acc={result["direction_of_effect"]["top50_degs_accuracy"]:.4f}')
    by_dataset = {}
    for ds, ds_inf in inf.get('by_dataset', {}).items():
        if len(ds_inf.get('pert_keys', [])) > 0:
            by_dataset[ds] = _compute_gene_level_analysis(ds_inf, n_genes, direction_threshold, pert_gene_masks=ds_inf.get('pert_gene_masks'))
    result['by_dataset'] = by_dataset
    return result


def _perturbation_retrieval(ctx, n_eval=200):
    '''Given desired outcome, can we find the right perturbation?'''
    inf = ctx.test_inference
    pert_keys = inf['pert_keys']
    mean_real_deltas = inf['mean_real_deltas']
    mean_control_states = inf['mean_control_states']
    pert_gene_masks = inf.get('pert_gene_masks', {})
    n_genes = ctx.config['num_genes']

    retrieval_banks = {}
    if ctx.seq_banks and 'dna' in ctx.seq_banks:
        retrieval_banks['dna'] = {'bank': ctx.seq_banks['dna'], 'mod_id': 0, 'mode': 0, 'use_seq': True}
    if ctx.seq_banks and 'chemical' in ctx.seq_banks:
        retrieval_banks['chemical'] = {'bank': ctx.seq_banks['chemical'], 'mod_id': 2, 'mode': 4, 'use_seq': True}
    if ctx.target_bank is not None:
        retrieval_banks['target_only'] = {'bank': ctx.target_bank, 'mod_id': 0, 'mode': 0, 'use_seq': False}

    if not retrieval_banks:
        return {'error': 'No perturbation banks available'}

    global_mean_control_total = inf['global_mean_control_total']

    def predict_all_deltas(control_x_np, bank_info, gene_mask=None, batch_size=64):
        bank, mod_id, mode_id, use_seq = bank_info['bank'], bank_info['mod_id'], bank_info['mode'], bank_info['use_seq']
        n_perts = bank.shape[0]
        control_x = torch.from_numpy(control_x_np).float().to(ctx.device)
        control_tot = torch.tensor(global_mean_control_total, device=ctx.device)
        if gene_mask is not None:
            unknown_mask = torch.from_numpy(~gene_mask).to(ctx.device)
        else:
            unknown_mask = None
        all_pred = []
        for start in range(0, n_perts, batch_size):
            end = min(start + batch_size, n_perts)
            B = end - start
            batch_idx = torch.arange(start, end, device=ctx.device)
            with torch.no_grad():
                control_batch = control_x.unsqueeze(0).expand(B, -1)
                control_tot_batch = control_tot.unsqueeze(0).expand(B)
                unk = unknown_mask.unsqueeze(0).expand(B, -1) if unknown_mask is not None else None
                z_ctx = ctx.biojepa.teacher(control_batch, control_tot_batch, mask_idx=None, unknown_mask=unk)
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
        for key in tqdm(eval_keys, desc=f'perturbation_retrieval ({bank_name})', disable=not ctx.config.get('verbose', True)):
            lookup_idx = key[idx_pos]
            if lookup_idx < 0 or lookup_idx >= n_bank:
                continue
            key_bank_info = {**bank_info, 'mode': key[3]}
            preds = predict_all_deltas(mean_control_states[key], key_bank_info, gene_mask=pert_gene_masks.get(key))
            gm = pert_gene_masks.get(key)
            if gm is not None:
                sims = cos_sim(preds[:, gm], mean_real_deltas[key][gm])
            else:
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


def _compute_uncertainty_calibration(inf, n_bins, sample_gene_masks=None):
    pred_deltas = inf['sample_pred_deltas']
    real_deltas = inf['sample_real_deltas']
    sample_logvars = inf['sample_logvars']
    pert_ids = inf['sample_pert_ids']
    target_ids = inf['sample_target_ids']
    pert_mods = inf['sample_pert_mods']
    pert_modes = inf['sample_all_mode'][:, 0] if 'sample_all_mode' in inf else np.zeros(len(pert_ids), dtype=int)
    cell_types = inf['sample_cell_types'] if 'sample_cell_types' in inf else np.zeros(len(pert_ids), dtype=int)

    if sample_gene_masks is not None:
        sq_err = (pred_deltas - real_deltas)**2 * sample_gene_masks
        sample_mse = sq_err.sum(axis=1) / sample_gene_masks.sum(axis=1).clip(1)
        sample_unc = (sample_logvars * sample_gene_masks).sum(axis=1) / sample_gene_masks.sum(axis=1).clip(1)
    else:
        sample_mse = np.mean((pred_deltas - real_deltas)**2, axis=1)
        sample_unc = sample_logvars.mean(axis=1)

    r, _ = pearsonr(sample_unc, sample_mse)
    pearson_r = 0.0 if np.isnan(r) else float(r)
    r, _ = spearmanr(sample_unc, sample_mse)
    spearman_r = 0.0 if np.isnan(r) else float(r)

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

    # Selective prediction curve
    sort_idx = np.argsort(sample_unc)
    selective = {}
    for pct in [25, 50, 75, 100]:
        n = max(1, int(len(sort_idx) * pct / 100))
        idx = sort_idx[:n]
        selective[f'top_{pct}pct'] = {'mse': float(np.mean(sample_mse[idx])), 'n_samples': n}

    n_perts = inf.get('sample_n_perts')
    pert_groups = defaultdict(list)
    for i in range(len(pert_ids)):
        if n_perts is not None and n_perts[i] > 1:
            continue
        key = (int(pert_ids[i]), int(target_ids[i]), int(pert_mods[i]), int(pert_modes[i]), int(cell_types[i]))
        pert_groups[key].append(i)

    pert_unc_arr = np.array([np.mean(sample_unc[pert_groups[p]]) for p in pert_groups])
    pert_err_arr = np.array([np.mean(sample_mse[pert_groups[p]]) for p in pert_groups])
    if len(pert_groups) >= 2:
        r, _ = pearsonr(pert_unc_arr, pert_err_arr)
        pert_pearson = 0.0 if np.isnan(r) else float(r)
        r, _ = spearmanr(pert_unc_arr, pert_err_arr)
        pert_spearman = 0.0 if np.isnan(r) else float(r)
    else:
        pert_pearson = 0.0
        pert_spearman = 0.0

    # Variance prediction R²
    variance_r2s = []
    for key, indices in pert_groups.items():
        if len(indices) < 3:
            continue
        idx = np.array(indices)
        gm = sample_gene_masks[idx].all(axis=0) if sample_gene_masks is not None else None
        if gm is not None:
            observed_var = np.var(real_deltas[idx][:, gm], axis=0)
            predicted_var = np.mean(np.exp(sample_logvars[idx][:, gm]), axis=0)
        else:
            observed_var = np.var(real_deltas[idx], axis=0)
            predicted_var = np.mean(np.exp(sample_logvars[idx]), axis=0)
        if np.std(observed_var) > 1e-9 and np.std(predicted_var) > 1e-9:
            variance_r2s.append(float(r2_score(observed_var, predicted_var)))

    result = {
        'config': {'samples': len(sample_mse), 'perturbations': len(pert_groups)},
        'sample_level': {'uncertainty_error_pearson': float(pearson_r), 'uncertainty_error_spearman': float(spearman_r), 'expected_calibration_error': float(ece), 'monotonicity_score': float(monotonicity)},
        'perturbation_level': {'uncertainty_error_pearson': float(pert_pearson), 'uncertainty_error_spearman': float(pert_spearman)},
        'bin_analysis': {'n_bins': n_bins, 'bin_mean_errors': [float(e) for e in bin_mean_error]},
        'selective_prediction': selective,
    }
    if variance_r2s:
        result['variance_prediction'] = {'r2_mean': float(np.mean(variance_r2s)), 'n_perturbations': len(variance_r2s)}
    return result


def _uncertainty_calibration(ctx, n_bins=10):
    '''Are confidence estimates meaningful?'''
    inf = ctx.test_inference
    sample_gene_masks = _build_sample_gene_masks(inf, ctx.dataset_gene_masks)
    result = _compute_uncertainty_calibration(inf, n_bins, sample_gene_masks=sample_gene_masks)
    print(f'uncertainty_calibration: ECE={result["sample_level"]["expected_calibration_error"]:.4f}, Monotonicity={result["sample_level"]["monotonicity_score"]:.2%}')
    by_dataset = {}
    ds_id_to_name = inf.get('dataset_id_to_name', {})
    ds_ids = inf.get('sample_dataset_ids')
    if ds_ids is not None:
        for ds_id, ds_name in ds_id_to_name.items():
            mask = ds_ids == ds_id
            if mask.sum() == 0:
                continue
            ds_inf = {
                'sample_pred_deltas': inf['sample_pred_deltas'][mask],
                'sample_real_deltas': inf['sample_real_deltas'][mask],
                'sample_logvars': inf['sample_logvars'][mask],
                'sample_pert_ids': inf['sample_pert_ids'][mask],
                'sample_target_ids': inf['sample_target_ids'][mask],
                'sample_pert_mods': inf['sample_pert_mods'][mask],
                'sample_all_mode': inf['sample_all_mode'][mask],
                'sample_cell_types': inf['sample_cell_types'][mask],
            }
            if 'sample_n_perts' in inf:
                ds_inf['sample_n_perts'] = inf['sample_n_perts'][mask]
            ds_sgm = None
            if ctx.dataset_gene_masks and ds_name in ctx.dataset_gene_masks:
                n_ds = int(mask.sum())
                ds_sgm = np.broadcast_to(ctx.dataset_gene_masks[ds_name], (n_ds, len(ctx.dataset_gene_masks[ds_name]))).copy()
            by_dataset[ds_name] = _compute_uncertainty_calibration(ds_inf, n_bins, sample_gene_masks=ds_sgm)
    result['by_dataset'] = by_dataset
    return result


def _action_vector_pathways(ctx):
    '''Do perturbations targeting same pathway produce similar action vectors?'''
    pathway_libs = ctx.pathway_annotations
    inf = ctx.alignment_inference
    gene_to_pathways = build_gene_to_pathways(pathway_libs['KEGG_2026'], min_pathway_size=15, max_pathway_size=300)

    results = {}

    if 'dna_actions' in inf:
        dna_actions = inf['dna_actions'].cpu().numpy()
        id_to_gene = ctx.id_to_gene
        valid_idx, valid_labels = [], []
        for pid, gene in id_to_gene.items():
            if gene in gene_to_pathways and pid < dna_actions.shape[0]:
                valid_idx.append(pid)
                valid_labels.append(gene_to_pathways[gene])
        if len(valid_idx) >= 10:
            metrics = compute_multilabel_pathway_similarity(dna_actions[valid_idx], valid_labels, min_pathway_size=5)
            results['dna'] = metrics
            print(f'action_vector_pathways DNA: ratio={metrics.get("similarity_ratio", "N/A")}')

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
                chem_pathway_map = {}
                for s, t in zip(chem_seq_idx, chem_target_idx):
                    if s < chem_actions.shape[0] and t in target_to_gene:
                        gene = target_to_gene[t]
                        if gene in gene_to_pathways:
                            s_int = int(s)
                            if s_int not in chem_pathway_map:
                                chem_pathway_map[s_int] = set()
                            chem_pathway_map[s_int].update(gene_to_pathways[gene])
                valid_idx = list(chem_pathway_map.keys())
                valid_labels = [list(chem_pathway_map[s]) for s in valid_idx]
                if len(valid_idx) >= 10:
                    metrics = compute_multilabel_pathway_similarity(chem_actions[valid_idx], valid_labels, min_pathway_size=5)
                    results['chemical'] = metrics
                    print(f'action_vector_pathways chemical: ratio={metrics.get("similarity_ratio", "N/A")}')

    if not results:
        return {'error': 'No action vectors available for pathway analysis'}

    return {'config': {'n_pathways_kegg': len(gene_to_pathways)}, 'by_modality': results}


def _compute_moa_matching(inf, gene_to_pathways, id_to_gene, target_to_gene, delta_key='mean_pred_deltas', pert_gene_masks=None):
    valid_keys, valid_labels = [], []
    seen_keys = set()
    for key in inf['pert_keys']:
        seq_idx, target_idx, modality = key[0], key[1], key[2]
        gene = None
        if modality == 0 and seq_idx >= 0 and seq_idx in id_to_gene:
            gene = id_to_gene[seq_idx]
        elif target_idx >= 0 and target_idx in target_to_gene:
            gene = target_to_gene[target_idx]
        if gene and gene in gene_to_pathways and key not in seen_keys:
            valid_keys.append(key)
            valid_labels.append(gene_to_pathways[gene])
            seen_keys.add(key)

    if len(valid_keys) < 4:
        return {'error': f'Not enough valid perturbations ({len(valid_keys)})'}

    deltas = inf.get(delta_key)
    if deltas is None:
        return {'error': f'{delta_key} not available'}

    if pert_gene_masks and delta_key != 'mean_latent_deltas':
        common_mask = np.ones(len(deltas[valid_keys[0]]), dtype=bool)
        for k in valid_keys:
            km = pert_gene_masks.get(k)
            if km is not None:
                common_mask &= km
        delta_matrix = np.array([deltas[k][common_mask] for k in valid_keys])
    else:
        delta_matrix = np.array([deltas[k] for k in valid_keys])
    return compute_multilabel_pathway_similarity(delta_matrix, valid_labels, min_pathway_size=3)


def _moa_matching(ctx):
    '''Do same-pathway perturbations produce similar predicted effects?'''
    inf = ctx.test_inference
    pathway_libs = ctx.pathway_annotations
    id_to_gene = ctx.id_to_gene
    gene_to_pathways = build_gene_to_pathways(pathway_libs['KEGG_2026'], min_pathway_size=15, max_pathway_size=300)

    target_to_gene = {}
    gene_to_target_path = ctx.paths['pert_dir'] / 'target_banks' / 'gene_to_target_idx.json'
    if gene_to_target_path.exists():
        with open(gene_to_target_path) as f:
            gene_to_target = json.load(f)
        target_to_gene = {tidx: gene.upper() for gene, tidx in gene_to_target.items() if not gene.startswith('ENSG')}

    expr = _compute_moa_matching(inf, gene_to_pathways, id_to_gene, target_to_gene, 'mean_pred_deltas', pert_gene_masks=inf.get('pert_gene_masks'))
    latent = _compute_moa_matching(inf, gene_to_pathways, id_to_gene, target_to_gene, 'mean_latent_deltas')

    if 'error' not in expr:
        print(f'moa_matching expression: Within={expr["mean_within_pathway"]:.4f}, Between={expr["mean_between_pathway"]:.4f}, Gap={expr["similarity_gap"]:.4f}, Ratio={expr["similarity_ratio"]:.4f}x')
    if 'error' not in latent:
        print(f'moa_matching latent: Within={latent["mean_within_pathway"]:.4f}, Between={latent["mean_between_pathway"]:.4f}, Gap={latent["similarity_gap"]:.4f}, Ratio={latent["similarity_ratio"]:.4f}x')

    by_dataset = {}
    for ds, ds_inf in inf.get('by_dataset', {}).items():
        if len(ds_inf.get('pert_keys', [])) > 0:
            by_dataset[ds] = {
                'expression': _compute_moa_matching(ds_inf, gene_to_pathways, id_to_gene, target_to_gene, 'mean_pred_deltas', pert_gene_masks=ds_inf.get('pert_gene_masks')),
                'latent': _compute_moa_matching(ds_inf, gene_to_pathways, id_to_gene, target_to_gene, 'mean_latent_deltas'),
            }

    return {'expression': expr, 'latent': latent, 'by_dataset': by_dataset}


def _compute_additive_baseline(combo_inf, single_deltas, single_gene_names, combo_to_genes, n_genes, gene_mask=None):
    gene_name_to_idx = {g: i for i, g in enumerate(single_gene_names)}
    model_mses, additive_mses, model_pearsons, additive_pearsons = [], [], [], []
    nonadd_model_mses, nonadd_pearsons = [], []
    per_key_results = {}

    for key in combo_inf['pert_keys']:
        genes = combo_to_genes.get(key)
        if genes is None or genes[0] is None or genes[1] is None:
            continue
        idx_a, idx_b = gene_name_to_idx.get(genes[0]), gene_name_to_idx.get(genes[1])
        if idx_a is None or idx_b is None:
            continue

        real_delta = combo_inf['mean_real_deltas'][key][:n_genes]
        pred_delta = combo_inf['mean_pred_deltas'][key][:n_genes]
        additive_delta = single_deltas[idx_a, :n_genes] + single_deltas[idx_b, :n_genes]
        additive_delta_full = additive_delta
        if gene_mask is not None:
            real_delta, pred_delta, additive_delta = real_delta[gene_mask], pred_delta[gene_mask], additive_delta[gene_mask]

        model_mse = float(np.mean((pred_delta - real_delta) ** 2))
        additive_mse = float(np.mean((additive_delta - real_delta) ** 2))
        model_mses.append(model_mse)
        additive_mses.append(additive_mse)

        if np.std(real_delta) > 1e-9 and np.std(pred_delta) > 1e-9:
            r = pearsonr(pred_delta, real_delta)[0]
            model_pearsons.append(0.0 if np.isnan(r) else float(r))
        if np.std(real_delta) > 1e-9 and np.std(additive_delta) > 1e-9:
            r = pearsonr(additive_delta, real_delta)[0]
            additive_pearsons.append(0.0 if np.isnan(r) else float(r))

        nonadd_deviation = np.abs(real_delta - additive_delta)
        top20_idx = np.argsort(nonadd_deviation)[-20:]
        nonadd_model_mses.append(float(np.mean((pred_delta[top20_idx] - real_delta[top20_idx]) ** 2)))
        real_sub, pred_sub = real_delta[top20_idx], pred_delta[top20_idx]
        if np.std(real_sub) > 1e-9 and np.std(pred_sub) > 1e-9:
            r = pearsonr(pred_sub, real_sub)[0]
            nonadd_pearsons.append(0.0 if np.isnan(r) else float(r))

        per_key_results[key] = {'genes': genes, 'model_mse': model_mse, 'additive_mse': additive_mse, 'additive_delta': additive_delta_full}

    if len(model_mses) == 0:
        skip = {'skipped': True, 'reason': 'no combos with both singles available'}
        return skip, skip, per_key_results

    additive_result = {
        'config': {'n_evaluated': len(model_mses), 'n_genes': n_genes},
        'model_mse': float(np.mean(model_mses)),
        'additive_mse': float(np.mean(additive_mses)),
        'model_pearson': float(np.mean(model_pearsons)) if model_pearsons else None,
        'additive_pearson': float(np.mean(additive_pearsons)) if additive_pearsons else None,
        'model_beats_additive_rate': float(np.mean([m < a for m, a in zip(model_mses, additive_mses)])),
    }

    nonadd_result = {
        'config': {'n_evaluated': len(nonadd_model_mses), 'n_genes_per_pert': 20},
        'mse': float(np.mean(nonadd_model_mses)),
        'pearson': float(np.mean(nonadd_pearsons)) if nonadd_pearsons else None,
    }

    return additive_result, nonadd_result, per_key_results


def _classify_generalization_splits(combo_to_genes, train_singles):
    split_map = {}
    for key, genes in combo_to_genes.items():
        if genes is None:
            continue
        n_unseen = sum(1 for g in genes if g not in train_singles)
        split_map[key] = n_unseen
    return split_map


def _combination_perturbation(ctx):
    '''Evaluate model performance on multi-perturbation samples (up to 4 perts, any mix of DNA/chemical).'''
    inf = ctx.test_inference
    n_genes = ctx.config['num_genes']

    n_perts_all = inf['sample_n_perts']
    combo_sample_mask = n_perts_all > 1
    if combo_sample_mask.sum() == 0:
        return {'skipped': True, 'reason': 'no multi-pert samples in test set'}

    combo_inf = {
        'pert_keys': inf['multi_pert_keys'],
        'mean_pred_deltas': inf['multi_pert_mean_pred_deltas'],
        'mean_real_deltas': inf['multi_pert_mean_real_deltas'],
        'mean_pred_abs': inf['multi_pert_mean_pred_abs'],
        'mean_real_abs': inf['multi_pert_mean_real_abs'],
        'mean_control_states': inf['multi_pert_mean_control_states'],
        'sample_mses': inf['sample_mses'][combo_sample_mask],
        'sample_correlations': inf['sample_correlations'][combo_sample_mask],
    }

    combo_n_perts = n_perts_all[combo_sample_mask]
    combo_modalities = inf['sample_all_modality'][combo_sample_mask]
    composition = {'by_n_perts': {}, 'modality_mix': defaultdict(int)}
    for n in sorted(set(combo_n_perts)):
        composition['by_n_perts'][int(n)] = int((combo_n_perts == n).sum())
    for i in range(len(combo_n_perts)):
        n = int(combo_n_perts[i])
        mods = tuple(sorted(int(combo_modalities[i, j]) for j in range(n)))
        mod_names = tuple('dna' if m == 0 else 'chem' if m == 2 else f'mod{m}' for m in mods)
        composition['modality_mix']['+'.join(mod_names)] += 1
    composition['modality_mix'] = dict(composition['modality_mix'])

    norman_mask = ctx.dataset_gene_masks.get('norman') if ctx.dataset_gene_masks else None
    combo_pgm = {k: norman_mask for k in combo_inf['pert_keys']} if norman_mask is not None else None
    expr_pred = _compute_expression_prediction(combo_inf, n_genes, pert_gene_masks=combo_pgm)

    combo_mapping = ctx.norman_combo_mapping
    single_deltas_data = ctx.norman_single_gene_deltas
    skip_reason = {'skipped': True, 'reason': 'norman_combo_mapping.json or norman_single_gene_deltas.npz not found'}
    combo_to_genes = {}
    if combo_mapping is None or single_deltas_data is None:
        additive_result, nonadd_result, per_key_results = skip_reason, skip_reason, {}
    else:
        seq_idx_map = inf['multi_pert_first_seq_idx']
        for key in combo_inf['pert_keys']:
            sid = seq_idx_map.get(key)
            genes = combo_mapping.get(str(sid)) if sid is not None else None
            combo_to_genes[key] = tuple(genes) if genes else None
        additive_result, nonadd_result, per_key_results = _compute_additive_baseline(
            combo_inf, single_deltas_data['deltas'], single_deltas_data['gene_names'], combo_to_genes, n_genes, gene_mask=norman_mask
        )

    gi_subtypes = ctx.norman_gi_subtypes
    if gi_subtypes is None or not per_key_results:
        gi_result = {'skipped': True, 'reason': 'norman_gi_subtypes.json not found or no additive baseline data'}
    else:
        name_to_subtype = {}
        for combo_name, subtype in gi_subtypes.items():
            name_to_subtype[combo_name] = subtype
            parts = combo_name.split('_')
            if len(parts) == 2:
                name_to_subtype[f'{parts[1]}_{parts[0]}'] = subtype

        by_subtype = defaultdict(lambda: {'model_mses': [], 'additive_mses': [], 'interaction_pearsons': []})
        for key, kres in per_key_results.items():
            combo_name = f'{kres["genes"][0]}_{kres["genes"][1]}'
            subtype = name_to_subtype.get(combo_name)
            if subtype is None:
                continue
            by_subtype[subtype]['model_mses'].append(kres['model_mse'])
            by_subtype[subtype]['additive_mses'].append(kres['additive_mse'])
            real_delta = combo_inf['mean_real_deltas'][key][:n_genes]
            pred_delta = combo_inf['mean_pred_deltas'][key][:n_genes]
            interaction_real = real_delta - kres['additive_delta']
            interaction_pred = pred_delta - kres['additive_delta']
            if norman_mask is not None:
                interaction_real, interaction_pred = interaction_real[norman_mask[:n_genes]], interaction_pred[norman_mask[:n_genes]]
            if np.std(interaction_real) > 1e-9 and np.std(interaction_pred) > 1e-9:
                r = pearsonr(interaction_pred, interaction_real)[0]
                by_subtype[subtype]['interaction_pearsons'].append(0.0 if np.isnan(r) else float(r))

        gi_result = {'config': {'n_subtypes': len(by_subtype)}, 'by_subtype': {}}
        for subtype, vals in sorted(by_subtype.items()):
            gi_result['by_subtype'][subtype] = {
                'n_combos': len(vals['model_mses']),
                'model_mse': float(np.mean(vals['model_mses'])),
                'additive_mse': float(np.mean(vals['additive_mses'])),
                'interaction_pearson': float(np.mean(vals['interaction_pearsons'])) if vals['interaction_pearsons'] else None,
            }

    ds_splits = ctx.dataset_splits
    if ds_splits is None or 'norman' not in ds_splits or not combo_to_genes:
        gen_splits = {'skipped': True, 'reason': 'dataset_splits.json not found or no combo gene mappings'}
    else:
        train_singles = {e for e in ds_splits['norman'].get('train', []) if '_' not in e}
        split_map = _classify_generalization_splits(combo_to_genes, train_singles)
        if not split_map:
            gen_splits = {'skipped': True, 'reason': 'no combos classifiable (combo_to_genes all None)'}
        else:
            gen_splits = {}
            for n_unseen in [0, 1, 2]:
                split_keys = [k for k, v in split_map.items() if v == n_unseen]
                if not split_keys:
                    continue
                split_mses, split_pearsons, split_mses_top20 = [], [], []
                for key in split_keys:
                    pred_d = combo_inf['mean_pred_deltas'][key][:n_genes]
                    real_d = combo_inf['mean_real_deltas'][key][:n_genes]
                    if norman_mask is not None:
                        pred_d, real_d = pred_d[norman_mask[:n_genes]], real_d[norman_mask[:n_genes]]
                    split_mses.append(float(np.mean((pred_d - real_d) ** 2)))
                    top20 = np.argsort(np.abs(real_d))[-20:]
                    split_mses_top20.append(float(np.mean((pred_d[top20] - real_d[top20]) ** 2)))
                    if np.std(pred_d) > 1e-9 and np.std(real_d) > 1e-9:
                        r = pearsonr(pred_d, real_d)[0]
                        split_pearsons.append(0.0 if np.isnan(r) else float(r))
                split_result = {
                    'n_combos': len(split_keys),
                    'mse': float(np.mean(split_mses)),
                    'mse_top20_degs': float(np.mean(split_mses_top20)),
                    'pearson_delta': float(np.mean(split_pearsons)) if split_pearsons else None,
                }
                add_keys = [k for k in split_keys if k in per_key_results]
                if add_keys:
                    m_mses = [per_key_results[k]['model_mse'] for k in add_keys]
                    a_mses = [per_key_results[k]['additive_mse'] for k in add_keys]
                    split_result['additive_baseline'] = {
                        'n_evaluated': len(add_keys),
                        'model_mse': float(np.mean(m_mses)),
                        'additive_mse': float(np.mean(a_mses)),
                        'model_beats_additive_rate': float(np.mean([m < a for m, a in zip(m_mses, a_mses)])),
                    }
                gen_splits[f'{n_unseen}/2_unseen'] = split_result

    n_additive = additive_result.get('config', {}).get('n_evaluated', 0) if not additive_result.get('skipped') else 0
    n_gi = sum(v['n_combos'] for v in gi_result.get('by_subtype', {}).values())
    n_gen = sum(v['n_combos'] for v in gen_splits.values() if isinstance(v, dict) and 'n_combos' in v)
    print(f'combination_perturbation: {len(combo_inf["pert_keys"])} combo perts, {int(combo_sample_mask.sum())} samples, '
          f'{n_additive} additive baseline, {n_gi} GI-labeled, {n_gen} generalization-classified')

    return {
        'expression_prediction': expr_pred,
        'n_combo_perturbations': len(combo_inf['pert_keys']),
        'n_combo_samples': int(combo_sample_mask.sum()),
        'composition': composition,
        'additive_baseline': additive_result,
        'non_additive_gene_mse': nonadd_result,
        'gi_subtype': gi_result,
        'generalization_splits': gen_splits,
    }


def _dose_response(ctx):
    '''Predicted vs real dose-response curves for chemical perturbations.'''
    inf = ctx.test_inference
    by_ds = inf.get('by_dataset', {})
    sciplex_inf = by_ds.get('sciplex')
    if not sciplex_inf or len(sciplex_inf.get('pert_keys', [])) == 0:
        return {'skipped': True, 'reason': 'no sciplex dataset in test set'}

    doses = inf['sample_doses']
    ds_id_to_name = inf.get('dataset_id_to_name', {})
    name_to_ds_id = {v: k for k, v in ds_id_to_name.items()}
    sciplex_ds_id = name_to_ds_id.get('sciplex')
    if sciplex_ds_id is None:
        return {'skipped': True, 'reason': 'sciplex dataset_id not found'}
    sciplex_mask = inf['sample_dataset_ids'] == sciplex_ds_id
    sciplex_gene_mask = ctx.dataset_gene_masks.get('sciplex') if ctx.dataset_gene_masks else None

    slot0_dose = doses[sciplex_mask, 0]
    valid_dose_mask = slot0_dose > 0
    if valid_dose_mask.sum() == 0:
        return {'skipped': True, 'reason': 'no dose data (dose=-1.0 for all samples)'}

    pred_deltas = inf['sample_pred_deltas'][sciplex_mask][valid_dose_mask]
    real_deltas = inf['sample_real_deltas'][sciplex_mask][valid_dose_mask]
    pert_ids = inf['sample_pert_ids'][sciplex_mask][valid_dose_mask]
    target_ids = inf['sample_target_ids'][sciplex_mask][valid_dose_mask]
    pert_mods = inf['sample_pert_mods'][sciplex_mask][valid_dose_mask]
    pert_modes = inf['sample_all_mode'][sciplex_mask][valid_dose_mask][:, 0]
    cell_types = inf['sample_cell_types'][sciplex_mask][valid_dose_mask]
    valid_doses = slot0_dose[valid_dose_mask]

    drug_dose_pred_severity, drug_dose_real_severity = defaultdict(list), defaultdict(list)
    for i in range(len(valid_doses)):
        key = (int(pert_ids[i]), int(target_ids[i]), int(pert_mods[i]), int(pert_modes[i]), int(cell_types[i]))
        d = float(valid_doses[i])
        pd_i = pred_deltas[i][sciplex_gene_mask] if sciplex_gene_mask is not None else pred_deltas[i]
        rd_i = real_deltas[i][sciplex_gene_mask] if sciplex_gene_mask is not None else real_deltas[i]
        drug_dose_pred_severity[key].append((d, float(np.linalg.norm(pd_i))))
        drug_dose_real_severity[key].append((d, float(np.linalg.norm(rd_i))))

    monotonic_count, real_monotonic_count, total_pairs = 0, 0, 0
    all_doses_flat, all_pred_sevs_flat, all_real_sevs_flat = [], [], []
    per_dose_pred_sevs, per_dose_real_sevs = defaultdict(list), defaultdict(list)
    curve_similarities = []

    for key in drug_dose_pred_severity:
        pred_dose_levels, real_dose_levels = defaultdict(list), defaultdict(list)
        for dose, sev in drug_dose_pred_severity[key]:
            pred_dose_levels[dose].append(sev)
        for dose, sev in drug_dose_real_severity[key]:
            real_dose_levels[dose].append(sev)
        sorted_doses = sorted(pred_dose_levels.keys())
        if len(sorted_doses) < 2:
            continue
        pred_mean_sevs = [float(np.mean(pred_dose_levels[d])) for d in sorted_doses]
        real_mean_sevs = [float(np.mean(real_dose_levels[d])) for d in sorted_doses]
        for i in range(len(sorted_doses) - 1):
            total_pairs += 1
            if pred_mean_sevs[i + 1] > pred_mean_sevs[i]:
                monotonic_count += 1
            if real_mean_sevs[i + 1] > real_mean_sevs[i]:
                real_monotonic_count += 1
        all_doses_flat.extend(sorted_doses)
        all_pred_sevs_flat.extend(pred_mean_sevs)
        all_real_sevs_flat.extend(real_mean_sevs)
        for d, ps, rs in zip(sorted_doses, pred_mean_sevs, real_mean_sevs):
            per_dose_pred_sevs[d].append(ps)
            per_dose_real_sevs[d].append(rs)
        if np.std(pred_mean_sevs) > 1e-9 and np.std(real_mean_sevs) > 1e-9:
            r = pearsonr(pred_mean_sevs, real_mean_sevs)[0]
            curve_similarities.append(0.0 if np.isnan(r) else float(r))

    if total_pairs == 0:
        return {'skipped': True, 'reason': 'no drugs with multiple dose levels'}

    r, _ = spearmanr(all_doses_flat, all_pred_sevs_flat)
    dose_severity_spearman = 0.0 if np.isnan(r) else float(r)
    r, _ = spearmanr(all_doses_flat, all_real_sevs_flat)
    real_dose_severity_spearman = 0.0 if np.isnan(r) else float(r)

    pred_vs_real_by_dose = {}
    for dose in sorted(per_dose_pred_sevs.keys()):
        pv, rv = per_dose_pred_sevs[dose], per_dose_real_sevs[dose]
        entry = {'n_drugs': len(pv), 'pearson': None}
        if len(pv) >= 3 and np.std(pv) > 1e-9 and np.std(rv) > 1e-9:
            r = pearsonr(pv, rv)[0]
            entry['pearson'] = 0.0 if np.isnan(r) else float(r)
        pred_vs_real_by_dose[float(dose)] = entry

    curve_sim = float(np.mean(curve_similarities)) if curve_similarities else None

    print(f'dose_response: monotonicity={monotonic_count/total_pairs:.2%}, '
          f'real_mono={real_monotonic_count/total_pairs:.2%}, spearman={dose_severity_spearman:.4f}, '
          f'curve_sim={curve_sim}')

    return {
        'config': {'n_drugs': len(drug_dose_pred_severity), 'n_valid_samples': int(valid_dose_mask.sum())},
        'monotonicity_score': float(monotonic_count / total_pairs),
        'real_monotonicity_score': float(real_monotonic_count / total_pairs),
        'dose_severity_spearman': float(dose_severity_spearman),
        'real_dose_severity_spearman': float(real_dose_severity_spearman),
        'n_dose_pairs': total_pairs,
        'curve_similarity': curve_sim,
        'pred_vs_real_by_dose': pred_vs_real_by_dose,
    }


# =============================================================================
# ALIGNMENT EVALS (v1.0 dual-path architecture)
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
        seq_emb = seq_bank[sample_idx].unsqueeze(1)
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
            dists = np.linalg.norm(mode_actions[m1] - mode_actions[m2], axis=1)
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


def _mode_semantic_consistency(ctx):
    '''Do learned mode embeddings reflect biological grouping (suppressive vs activating)?'''
    seq_bank = ctx.seq_banks.get('dna') if ctx.seq_banks else None
    if seq_bank is None:
        return {'error': 'No DNA sequence bank available'}

    SUPPRESSIVE = {'crispri': 0, 'knockout': 3, 'inhibitor': 4, 'degrader': 6}
    ACTIVATING = {'crispra': 1, 'overexpression': 2, 'agonist': 5}
    ALL_MODES = {**SUPPRESSIVE, **ACTIVATING}

    n_sample = min(200, seq_bank.shape[0])
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(seq_bank.shape[0], n_sample, replace=False)

    mode_vectors = {}
    with torch.no_grad():
        seq_emb = seq_bank[sample_idx].unsqueeze(1)
        modality_ids = torch.zeros(n_sample, 1, dtype=torch.long, device=ctx.device)
        pert_mask = torch.ones(n_sample, 1, dtype=torch.bool, device=ctx.device)
        for mode_name, mode_id in ALL_MODES.items():
            mode_ids = torch.full((n_sample, 1), mode_id, dtype=torch.long, device=ctx.device)
            actions = ctx.biojepa.composer.encode_sequence_path(seq_emb, modality_ids, mode_ids, pert_mask)
            v = actions.squeeze(1).cpu().numpy()
            norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
            mode_vectors[mode_name] = v / norms

    mode_list = list(ALL_MODES.keys())
    pairwise_cosine = {}
    for i, m1 in enumerate(mode_list):
        for m2 in mode_list[i+1:]:
            sim = float(np.mean(np.sum(mode_vectors[m1] * mode_vectors[m2], axis=1)))
            pairwise_cosine[f'{m1}_vs_{m2}'] = sim

    suppressive_names = set(SUPPRESSIVE.keys())
    activating_names = set(ACTIVATING.keys())
    within_sims, cross_sims = [], []
    within_supp, within_act = [], []
    for i, m1 in enumerate(mode_list):
        for m2 in mode_list[i+1:]:
            sim = pairwise_cosine[f'{m1}_vs_{m2}']
            same_group = (m1 in suppressive_names and m2 in suppressive_names) or (m1 in activating_names and m2 in activating_names)
            if same_group:
                within_sims.append(sim)
                if m1 in suppressive_names:
                    within_supp.append(sim)
                else:
                    within_act.append(sim)
            else:
                cross_sims.append(sim)

    within_group_avg = float(np.mean(within_sims))
    cross_group_avg = float(np.mean(cross_sims))
    semantic_gap = float(within_group_avg - cross_group_avg)

    result = {
        'embedding_semantics': {
            'config': {'n_samples': int(n_sample), 'n_modes': len(ALL_MODES)},
            'pairwise_cosine': pairwise_cosine,
            'within_suppressive_sim': float(np.mean(within_supp)),
            'within_activating_sim': float(np.mean(within_act)),
            'within_group_avg': within_group_avg,
            'cross_group_avg': cross_group_avg,
            'semantic_gap': semantic_gap,
        }
    }

    pairs = ctx.alignment_pairs
    if pairs is None:
        result['cross_mode_retrieval'] = {'skipped': True, 'reason': 'no alignment pairs'}
        print(f'mode_semantic_consistency: semantic_gap={semantic_gap:.4f}')
        return result

    dna_mask = pairs['modality'] == 0
    dna_seq = pairs['seq_idx'][dna_mask]
    dna_target = pairs['target_idx'][dna_mask]
    dna_mode = pairs['mode'][dna_mask]

    target_to_crispri = defaultdict(set)
    target_to_crispra = defaultdict(set)
    for s, t, m in zip(dna_seq, dna_target, dna_mode):
        if s < seq_bank.shape[0]:
            if m == 0:
                target_to_crispri[int(t)].add(int(s))
            elif m == 1:
                target_to_crispra[int(t)].add(int(s))

    matched_targets = sorted(set(target_to_crispri) & set(target_to_crispra))
    if len(matched_targets) < 5:
        result['cross_mode_retrieval'] = {'skipped': True, 'reason': f'insufficient cross-mode genes ({len(matched_targets)})'}
        print(f'mode_semantic_consistency: semantic_gap={semantic_gap:.4f}')
        return result

    crispri_vecs, crispra_vecs = [], []
    with torch.no_grad():
        modality_ids_1 = torch.zeros(1, 1, dtype=torch.long, device=ctx.device)
        pert_mask_1 = torch.ones(1, 1, dtype=torch.bool, device=ctx.device)
        mode_i = torch.zeros(1, 1, dtype=torch.long, device=ctx.device)
        mode_a = torch.ones(1, 1, dtype=torch.long, device=ctx.device)

        for t in matched_targets:
            i_indices = sorted(target_to_crispri[t])
            a_indices = sorted(target_to_crispra[t])

            i_embs = seq_bank[i_indices].unsqueeze(1)
            i_mod = modality_ids_1.expand(len(i_indices), 1)
            i_mask = pert_mask_1.expand(len(i_indices), 1)
            i_mode = mode_i.expand(len(i_indices), 1)
            i_actions = ctx.biojepa.composer.encode_sequence_path(i_embs, i_mod, i_mode, i_mask)
            i_avg = i_actions.squeeze(1).mean(dim=0, keepdim=True)
            crispri_vecs.append(F.normalize(i_avg, dim=1))

            a_embs = seq_bank[a_indices].unsqueeze(1)
            a_mod = modality_ids_1.expand(len(a_indices), 1)
            a_mask = pert_mask_1.expand(len(a_indices), 1)
            a_mode = mode_a.expand(len(a_indices), 1)
            a_actions = ctx.biojepa.composer.encode_sequence_path(a_embs, a_mod, a_mode, a_mask)
            a_avg = a_actions.squeeze(1).mean(dim=0, keepdim=True)
            crispra_vecs.append(F.normalize(a_avg, dim=1))

    crispri_mat = torch.cat(crispri_vecs, dim=0)
    crispra_mat = torch.cat(crispra_vecs, dim=0)

    matched_sim = float(F.cosine_similarity(crispri_mat, crispra_mat, dim=1).mean().item())
    n_genes = crispri_mat.shape[0]
    n_random = min(1000, n_genes * n_genes)
    random_sims = []
    for _ in range(n_random):
        i_idx = rng.randint(0, n_genes)
        a_idx = rng.randint(0, n_genes)
        random_sims.append(float(F.cosine_similarity(crispri_mat[i_idx:i_idx+1], crispra_mat[a_idx:a_idx+1], dim=1).item()))
    random_mean = float(np.mean(random_sims))

    sim_matrix = torch.mm(crispri_mat, crispra_mat.T).cpu().numpy()
    K_VALUES = [1, 5]

    def _retrieval_metrics(sim_mat):
        n = sim_mat.shape[0]
        reciprocal_ranks = []
        recall_at_k = {k: [] for k in K_VALUES}
        for row in range(n):
            sorted_indices = np.argsort(sim_mat[row])[::-1]
            for rank, idx in enumerate(sorted_indices, 1):
                if idx == row:
                    reciprocal_ranks.append(1.0 / rank)
                    for k in K_VALUES:
                        recall_at_k[k].append(1 if rank <= k else 0)
                    break
        return {
            'mrr': float(np.mean(reciprocal_ranks)),
            'recall_at_1': float(np.mean(recall_at_k[1])),
            'recall_at_5': float(np.mean(recall_at_k[5])),
        }

    retrieval_i_to_a = _retrieval_metrics(sim_matrix)
    retrieval_a_to_i = _retrieval_metrics(sim_matrix.T)
    cross_mode_mrr = float((retrieval_i_to_a['mrr'] + retrieval_a_to_i['mrr']) / 2)

    result['cross_mode_retrieval'] = {
        'config': {'n_matched_genes': int(n_genes)},
        'similarity': {'matched_mean': matched_sim, 'random_mean': random_mean, 'gap': float(matched_sim - random_mean)},
        'retrieval_i_to_a': retrieval_i_to_a,
        'retrieval_a_to_i': retrieval_a_to_i,
        'cross_mode_mrr': cross_mode_mrr,
    }

    print(f'mode_semantic_consistency: semantic_gap={semantic_gap:.4f}, cross_mode_mrr={cross_mode_mrr:.4f}')
    return result


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
                seq_emb[i, 0] = dna_bank[s]

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
                seq_emb[i, 0] = dna_bank[s]

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
        test_loader = TrainingLoader(batch_size=ctx.config['batch_size'], split=ctx.config.get('eval_split', 'test'), data_dir=ctx.paths['train_dir'], device=ctx.device, seed=ctx.config.get('seed', 1337))
    except RuntimeError as e:
        return {'error': f'Could not load test shards: {e}. Run data_prep_03_shards.ipynb first.'}

    test_steps = min(500, ctx.config.get('test_total_examples', test_loader.total_samples) // ctx.config['batch_size'])
    single_pert_samples, multi_pert_samples = [], []

    for _ in tqdm(range(test_steps), desc='multi_pert_alignment: Scanning for multi-pert', disable=not ctx.config.get('verbose', True)):
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
            seq_emb = dna_bank[s].unsqueeze(0).unsqueeze(0)
            target_emb = target_bank[t].unsqueeze(0).unsqueeze(0)
            mod = torch.zeros(1, 1, dtype=torch.long, device=ctx.device)
            mode = torch.full((1, 1), m, dtype=torch.long, device=ctx.device)
            mask = torch.ones(1, 1, dtype=torch.bool, device=ctx.device)
            seq_action = ctx.biojepa.composer.encode_sequence_path(seq_emb, mod, mode, mask)
            target_action = ctx.biojepa.composer.encode_target_path(target_emb, mode, mask)
            seq_pooled = seq_action.squeeze(1)
            target_pooled = target_action.squeeze(1)
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
                seq_emb[0, j] = dna_bank[s]
                target_emb[0, j] = target_bank[t]
                mode_ids[0, j] = m
            mod = torch.zeros(1, n, dtype=torch.long, device=ctx.device)
            mask = torch.ones(1, n, dtype=torch.bool, device=ctx.device)
            seq_action = ctx.biojepa.composer.encode_sequence_path(seq_emb, mod, mode_ids, mask)
            target_action = ctx.biojepa.composer.encode_target_path(target_emb, mode_ids, mask)
            seq_pooled = seq_action.mean(dim=1)
            target_pooled = target_action.mean(dim=1)
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
                    seq_emb[i, 0] = dna_bank[p]
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


def _cross_modality_alignment(ctx):
    '''Do DNA and chemical perturbations targeting the same gene produce similar action vectors?'''
    inf = ctx.alignment_inference
    pairs = ctx.alignment_pairs

    if 'dna_actions_norm' not in inf or 'chem_actions_norm' not in inf:
        return {'skipped': True, 'reason': 'need both dna_actions_norm and chem_actions_norm'}

    dna_norm = inf['dna_actions_norm']
    chem_norm = inf['chem_actions_norm']

    dna_mask = pairs['modality'] == 0
    chem_mask = pairs['modality'] == 2
    target_to_dna = defaultdict(set)
    target_to_chem = defaultdict(set)
    for s, t in zip(pairs['seq_idx'][dna_mask], pairs['target_idx'][dna_mask]):
        if s < dna_norm.shape[0]:
            target_to_dna[int(t)].add(int(s))
    for s, t in zip(pairs['seq_idx'][chem_mask], pairs['target_idx'][chem_mask]):
        if s < chem_norm.shape[0]:
            target_to_chem[int(t)].add(int(s))

    matched_targets = [t for t in target_to_dna if t in target_to_chem]
    if len(matched_targets) < 5:
        return {'skipped': True, 'reason': f'insufficient cross-modality pairs ({len(matched_targets)} matched targets, need 5)'}

    matched_sims, random_sims = [], []
    rng = np.random.RandomState(42)
    for t in matched_targets:
        for dna_idx in target_to_dna[t]:
            for chem_idx in target_to_chem[t]:
                sim = torch.dot(dna_norm[dna_idx], chem_norm[chem_idx]).item()
                matched_sims.append(sim)
    all_chem_indices = list({s for seqs in target_to_chem.values() for s in seqs})
    all_dna_indices = list({s for seqs in target_to_dna.values() for s in seqs})
    n_random = min(5000, len(all_dna_indices) * len(all_chem_indices))
    for _ in range(n_random):
        d = rng.choice(all_dna_indices)
        c = rng.choice(all_chem_indices)
        sim = torch.dot(dna_norm[d], chem_norm[c]).item()
        random_sims.append(sim)

    K_VALUES = [1, 5, 10]
    all_chem_vecs = chem_norm[all_chem_indices]
    reciprocal_ranks = []
    recall_at_k = {k: [] for k in K_VALUES}
    for t in matched_targets:
        correct_chem = target_to_chem[t]
        for dna_idx in target_to_dna[t]:
            sims = torch.mv(all_chem_vecs, dna_norm[dna_idx]).cpu().numpy()
            sorted_indices = np.argsort(sims)[::-1]
            for rank, idx in enumerate(sorted_indices, 1):
                if all_chem_indices[idx] in correct_chem:
                    reciprocal_ranks.append(1.0 / rank)
                    for k in K_VALUES:
                        recall_at_k[k].append(1 if rank <= k else 0)
                    break

    result = {
        'config': {'n_matched_targets': len(matched_targets), 'n_matched_pairs': len(matched_sims), 'n_dna_queries': len(reciprocal_ranks)},
        'cosine_similarity': {'matched_mean': float(np.mean(matched_sims)), 'random_mean': float(np.mean(random_sims)), 'gap': float(np.mean(matched_sims) - np.mean(random_sims))},
    }
    if reciprocal_ranks:
        result['retrieval'] = {'mrr': float(np.mean(reciprocal_ranks)), 'recall_at_k': {str(k): float(np.mean(recall_at_k[k])) for k in K_VALUES}}
    print(f'cross_modality_alignment: {len(matched_targets)} matched targets, matched_sim={np.mean(matched_sims):.4f}, random_sim={np.mean(random_sims):.4f}')
    return result
