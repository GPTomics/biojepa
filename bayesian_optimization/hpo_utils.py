'''HPO utility functions for BioJEPA v0.6 pretraining optimization.'''

import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from dataloader_v0_6 import PretrainBatch


def is_valid_config(params):
    '''Validate HPO configuration constraints.'''
    head_dim = params['embed_dim'] // params['heads']
    if head_dim < 32 or head_dim > 128:
        return False
    if params['embed_dim'] % params['heads'] != 0:
        return False
    if params['embed_dim'] >= 512 and params['lr'] > 1.5e-3:
        return False
    if params['embed_dim'] >= 512 and head_dim > 96:
        return False
    return True


def compute_step_budget(embed_dim):
    '''Step budget scales with model capacity.'''
    return int(25000 + 8000 * (embed_dim / 256 - 1))


def derive_parameters(params):
    '''Derive dependent parameters from HPO params.'''
    return {
        'n_pre_layer': max(1, round(params['n_pre_layer_ratio'] * params['n_layer'])),
        'head_dim': params['embed_dim'] // params['heads'],
    }


def check_identity_shortcut_cheap(eval_results):
    if 'reconstruction' not in eval_results or 'metrics' not in eval_results.get('reconstruction', {}):
        return True, 'reconstruction eval missing or failed'
    recon = eval_results['reconstruction']['metrics'].get('pearson_r', 0)
    if recon > 0.95:
        return True, f'Identity shortcut (reconstruction): pearson={recon:.3f}'
    return False, None


def check_identity_shortcut_full(eval_results):
    if 'gene_embedding_pathways' not in eval_results:
        return True, 'gene_embedding_pathways eval missing'
    kegg = eval_results['gene_embedding_pathways'].get('kegg', {})
    pathway = kegg.get('silhouette_score', -1.0)
    if pathway < 0.025:
        return True, f'Identity shortcut (pathway collapse): silhouette={pathway:.3f}'
    return False, None


def check_cell_type_collapse(eval_results):
    if 'cell_type_probing' not in eval_results:
        return False, None
    cell_type = eval_results['cell_type_probing']
    if 'error' in cell_type:
        return False, None
    metrics = cell_type.get('metrics', {})
    if not metrics:
        return True, 'cell_type_probing metrics missing'
    accuracy = metrics.get('accuracy', 0)
    chance = metrics.get('chance', 1.0)
    n_types = cell_type.get('config', {}).get('num_cell_types', 2)
    if accuracy < chance * 1.5 and n_types >= 2:
        return True, f'Cell type collapse: acc={accuracy:.3f}, chance={chance:.3f}'
    return False, None


def check_vicreg_collapse(eval_results, embed_dim):
    if 'latent_space_health' not in eval_results:
        return True, 'latent_space_health eval missing'
    health = eval_results['latent_space_health']
    variance_dict = health.get('variance', {})
    dead_dims = variance_dict.get('n_dead_dims', embed_dim)
    if dead_dims > 0.02 * embed_dim:
        return True, f'Dead dimension collapse: {dead_dims}/{embed_dim} dims dead'
    eff_dim_dict = health.get('effective_dimensionality', {})
    eff_dim = eff_dim_dict.get('90_percent', 0)
    if eff_dim / embed_dim < 0.2:
        return True, f'Dimensional collapse: eff_dim={eff_dim}, embed_dim={embed_dim}'
    return False, None


def check_perturbation_signal(eval_results):
    if 'perturbation_detection' not in eval_results:
        return True, 'perturbation_detection eval missing'
    metrics = eval_results['perturbation_detection'].get('metrics', {})
    auroc = metrics.get('auroc', 0)
    if auroc < 0.60:
        return True, f'Perturbation signal lost: auroc={auroc:.3f}'
    return False, None


# Objective computation

def compute_health_score(health, embed_dim):
    '''Compute latent space health score.'''
    eff_dim_dict = health.get('effective_dimensionality', {})
    eff_dim = eff_dim_dict.get('90_percent', embed_dim // 2)
    eff_dim_pct = eff_dim / embed_dim

    variance_dict = health.get('variance', {})
    dead_dims = variance_dict.get('n_dead_dims', 0)

    dim_score = max(0, min(1, (eff_dim_pct - 0.3) / 0.5))
    dead_pct = dead_dims / embed_dim
    dead_score = max(0, 1 - dead_pct / 0.02)
    return float(0.7 * dim_score + 0.3 * dead_score)


def compute_pathway_score(pathway_results):
    '''Compute gene embedding pathway score.'''
    sil = pathway_results.get('kegg', {}).get('silhouette_score', -0.05)
    return float(min(1.0, max(0, (sil + 0.05) / 0.15)))


def compute_consistency_score(consistency_results):
    '''Compute embedding consistency score.'''
    ratio = consistency_results.get('metrics', {}).get('inter_intra_ratio', 1.0)
    return float(min(1.0, max(0, (ratio - 1.0) / 1.0)))


def compute_objective(eval_results, embed_dim):
    '''Compute full objective score (for pruner reporting).'''
    weights = {
        'perturbation_detection': 0.15,
        'batch_invariance': 0.15,
        'reconstruction': 0.15,
        'latent_space_health': 0.15,
        'gene_embedding_pathways': 0.20,
        'essential_gene_prediction': 0.10,
        'embedding_consistency': 0.10,
    }

    normalized = {
        'perturbation_detection': max(0, (eval_results.get('perturbation_detection', {}).get('metrics', {}).get('auroc', 0.5) - 0.5) * 2),
        'batch_invariance': min(1.0, eval_results.get('batch_invariance', {}).get('invariance_ratio', 0) / 5.0),
        'reconstruction': eval_results.get('reconstruction', {}).get('metrics', {}).get('pearson_r', 0),
        'latent_space_health': compute_health_score(eval_results.get('latent_space_health', {}), embed_dim),
        'gene_embedding_pathways': compute_pathway_score(eval_results.get('gene_embedding_pathways', {})),
        'essential_gene_prediction': max(0, (eval_results.get('essential_gene_prediction', {}).get('classification', {}).get('auroc_test', 0.5) - 0.5) * 2),
        'embedding_consistency': compute_consistency_score(eval_results.get('embedding_consistency', {})),
    }

    return float(sum(weights[k] * normalized[k] for k in weights))


# Data subsampling

def create_phase1_subsample(shard_dir, fraction=0.5, seed=42, balance='equal'):
    '''Create balanced subsample of pretraining shards for Phase 1.

    Args:
        shard_dir: Path to pretraining directory
        fraction: Fraction of smallest dataset to use
        seed: Random seed for reproducibility
        balance: 'equal' for same shards per dataset, 'proportional' for weighted

    Returns:
        List of Path objects to selected shards
    '''
    rng = np.random.default_rng(seed)
    shards_by_dataset = defaultdict(list)

    shard_dir = Path(shard_dir)
    train_dir = shard_dir / 'train'
    for shard in train_dir.glob('pt_*_train_*.npz'):
        parts = shard.stem.split('_')
        if 'train' not in parts:
            continue
        train_idx = parts.index('train')
        dataset = '_'.join(parts[1:train_idx])
        shards_by_dataset[dataset].append(shard)

    if not shards_by_dataset:
        raise RuntimeError(f'No shards found in {train_dir}')

    if balance == 'equal':
        min_shards = min(len(s) for s in shards_by_dataset.values())
        n_per_dataset = max(1, int(min_shards * fraction))
        selected = []
        for dataset, shards in sorted(shards_by_dataset.items()):
            need_replacement = n_per_dataset > len(shards)
            chosen = rng.choice(shards, n_per_dataset, replace=need_replacement)
            selected.extend(chosen)
    else:
        min_per_dataset = 5
        selected = []
        for dataset, shards in sorted(shards_by_dataset.items()):
            n_select = max(min_per_dataset, int(len(shards) * fraction))
            n_select = min(n_select, len(shards))
            chosen = rng.choice(shards, n_select, replace=False)
            selected.extend(chosen)

    return selected


# Subsampled loader

class SubsampledPretrainLoader:
    '''PretrainLoader that uses a specific list of shard paths.'''

    def __init__(self, batch_size, shard_paths, device, seed=42):
        self.batch_size = batch_size
        self.device = device
        self.shards = list(shard_paths)
        self.rng = np.random.default_rng(seed)

        if not self.shards:
            raise RuntimeError('No shards provided')

        self.remaining_shards = []
        self.current_data = None
        self.perm = None
        self.current_position = 0
        self.total_samples_in_shard = 0

        with np.load(self.shards[0]) as data:
            samples_per_shard = len(data['x'])
        self.total_samples = samples_per_shard * len(self.shards)

        self.reset()

    def reset(self):
        self.remaining_shards = list(self.shards)
        self.rng.shuffle(self.remaining_shards)
        self._load_next_shard()

    def _load_next_shard(self):
        max_attempts = len(self.shards) + 1
        for _ in range(max_attempts):
            if not self.remaining_shards:
                self.remaining_shards = list(self.shards)
                self.rng.shuffle(self.remaining_shards)

            shard = self.remaining_shards.pop()
            with np.load(shard) as data:
                self.current_data = (
                    data['x'].astype(np.float32),
                    data['total'].astype(np.float32)
                )

            n_samples = len(self.current_data[0])
            if n_samples > 0:
                self.perm = self.rng.permutation(n_samples)
                self.current_position = 0
                self.total_samples_in_shard = n_samples
                return

        raise RuntimeError(f'All {len(self.shards)} shards are empty')

    def next_batch(self):
        batch_x, batch_total = [], []
        samples_needed = self.batch_size

        while samples_needed > 0:
            remaining_in_shard = self.total_samples_in_shard - self.current_position
            if remaining_in_shard == 0:
                self._load_next_shard()
                remaining_in_shard = self.total_samples_in_shard

            take = min(samples_needed, remaining_in_shard)
            indices = self.perm[self.current_position:self.current_position + take]
            self.current_position += take
            samples_needed -= take

            batch_x.append(self.current_data[0][indices])
            batch_total.append(self.current_data[1][indices])

        x = torch.from_numpy(np.concatenate(batch_x)).to(self.device)
        total = torch.from_numpy(np.concatenate(batch_total)).to(self.device)
        return PretrainBatch(x=x, total=total)


# Eval runner

def run_selected_evals(ctx, eval_names):
    '''Run only the specified evals.

    Args:
        ctx: EvalContext with loaded model
        eval_names: List of eval names to run

    Returns:
        Dict mapping eval_name -> results
    '''
    from evals.evals import (
        _batch_invariance, _perturbation_detection, _latent_space_health,
        _reconstruction, _gene_embedding_pathways, _essential_gene_prediction,
        _embedding_consistency, _cell_type_probing
    )

    EVAL_FUNCTIONS = {
        'batch_invariance': _batch_invariance,
        'perturbation_detection': _perturbation_detection,
        'latent_space_health': _latent_space_health,
        'reconstruction': _reconstruction,
        'gene_embedding_pathways': _gene_embedding_pathways,
        'essential_gene_prediction': _essential_gene_prediction,
        'embedding_consistency': _embedding_consistency,
        'cell_type_probing': _cell_type_probing,
    }

    results = {}
    for name in eval_names:
        if name in EVAL_FUNCTIONS:
            results[name] = EVAL_FUNCTIONS[name](ctx)
    return results
