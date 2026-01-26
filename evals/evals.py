'''BioJEPA v0.6 Evaluation Suite

Two entry points:
- run_pretraining_evals(ctx): batch_invariance, gene_embedding_pathways, essential_gene_prediction
- run_full_model_evals(ctx): expression_prediction, gene_level_analysis, perturbation_retrieval,
                             uncertainty_calibration, action_vector_pathways, moa_matching
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

import biojepa_v0_6 as model
from dataloader_v0_6 import TrainingLoader
from .linear_expression_decoder import BenchmarkDecoder, BenchmarkDecoderConfig
from .pathway_utils import load_pathway_annotations, map_genes_to_pathways, compute_pathway_clustering_metrics
from .linear_classifier import train_linear_classifier


DEFAULT_CONFIG = {
    'batch_size': 32,
    'num_genes': 5000,
    'n_layer': 2,
    'heads': 2,
    'embed_dim': 8,
    'pert_latent_dim': 320,
    'pert_mode_dim': 64,
    'test_total_examples': 38829
}


class EvalContext:
    '''Unified context for running evaluations. All loading is lazy.'''

    def __init__(self, config=None, data_root='/Users/djemec/data/jepa/v0_6', checkpoint_root='/Users/djemec/data/jepa/v0_6'):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.paths = self._get_paths(data_root, checkpoint_root)
        print(f'Using {self.device}')

        self._biojepa = None
        self._decoder = None
        self._input_bank = None
        self._gene_embeddings = None
        self._gene_names = None
        self._test_inference = None

    @classmethod
    def from_trained_model(cls, biojepa_model, decoder=None, data_root='/Users/djemec/data/jepa/v0_6', config=None):
        '''Create context from in-memory trained model (for notebook integration).'''
        ctx = cls(config=config, data_root=data_root, checkpoint_root=data_root)
        ctx._biojepa = biojepa_model
        ctx._biojepa.freeze_encoders()
        ctx._biojepa.eval()
        for param in ctx._biojepa.parameters():
            param.requires_grad = False
        if decoder is not None:
            ctx._decoder = decoder
            ctx._decoder.eval()
        return ctx

    def _get_paths(self, data_root, checkpoint_root):
        data_dir = Path(data_root)
        return {
            'data_dir': data_dir,
            'train_dir': data_dir / 'training',
            'checkpoint_dir': Path(checkpoint_root) / 'checkpoints',
            'pert_dir': data_dir / 'pert_embd'
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
    def input_bank(self):
        if self._input_bank is None:
            self._input_bank = torch.from_numpy(np.load(self.paths['pert_dir'] / 'input_embeddings_dna.npy')).float().to(self.device)
            print(f'Loaded input bank: {self._input_bank.shape}')
        return self._input_bank

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
        sample_pred_deltas, sample_real_deltas, sample_logvars, sample_pert_ids = [], [], [], []
        sample_mses, sample_correlations = [], []

        for _ in tqdm(range(test_steps), desc='Running test inference'):
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

            pred_delta_np, real_delta_np = pred_delta.cpu().numpy(), real_delta.cpu().numpy()
            pred_abs_np, real_abs_np = pred_abs.cpu().numpy(), case_x.cpu().numpy()
            logvar_np = z_pred_logvar.mean(dim=-1).cpu().numpy()
            p_idx_np = p_idx.cpu().numpy().flatten()
            cont_x_np = cont_x.cpu().numpy()

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
                bulk_control_states[pid].append(cont_x_np[i])

                sample_mses.append(np.mean((pred_delta_np[i] - real_delta_np[i])**2))
                top_20_idx = np.argsort(np.abs(real_delta_np[i]))[-20:]
                p_top, t_top = pred_delta_np[i][top_20_idx], real_delta_np[i][top_20_idx]
                if np.std(p_top) > 1e-9 and np.std(t_top) > 1e-9:
                    corr, _ = pearsonr(p_top, t_top)
                    sample_correlations.append(0.0 if np.isnan(corr) else corr)
                else:
                    sample_correlations.append(0.0)

        pert_ids = list(bulk_pred_deltas.keys())
        print(f'Aggregated {len(pert_ids)} perturbations, {len(sample_mses)} samples')

        return {
            'pert_ids': pert_ids,
            'mean_pred_deltas': {pid: np.mean(np.stack(bulk_pred_deltas[pid]), axis=0) for pid in pert_ids},
            'mean_real_deltas': {pid: np.mean(np.stack(bulk_real_deltas[pid]), axis=0) for pid in pert_ids},
            'mean_pred_abs': {pid: np.mean(np.stack(bulk_pred_abs[pid]), axis=0) for pid in pert_ids},
            'mean_real_abs': {pid: np.mean(np.stack(bulk_real_abs[pid]), axis=0) for pid in pert_ids},
            'mean_control_states': {pid: np.mean(np.stack(bulk_control_states[pid]), axis=0) for pid in pert_ids},
            'sample_mses': np.array(sample_mses),
            'sample_correlations': np.array(sample_correlations),
            'sample_pred_deltas': np.concatenate(sample_pred_deltas, axis=0),
            'sample_real_deltas': np.concatenate(sample_real_deltas, axis=0),
            'sample_logvars': np.concatenate(sample_logvars, axis=0),
            'sample_pert_ids': np.concatenate(sample_pert_ids, axis=0)
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


# =============================================================================
# PRETRAINING EVALS
# =============================================================================

def _batch_invariance(ctx):
    '''Are representations confounded by batch effects?'''
    test_loader = TrainingLoader(batch_size=ctx.config['batch_size'], split='test', data_dir=ctx.paths['train_dir'], device=ctx.device, return_batch_id=True)
    test_steps = ctx.config['test_total_examples'] // ctx.config['batch_size']

    all_emb, all_batch, all_pert = [], [], []
    with torch.no_grad():
        for _ in tqdm(range(test_steps), desc='batch_invariance: Extracting embeddings'):
            cont_x, cont_tot, case_x, case_tot, p_idx, p_mod, p_mode, batch_id = test_loader.next_batch()
            all_emb.append(ctx.biojepa.student(cont_x, cont_tot, mask_idx=None).mean(dim=1).cpu().numpy())
            all_batch.append(batch_id.cpu().numpy())
            all_pert.append(p_idx.cpu().numpy())

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
    pathway_libs = load_pathway_annotations(['KEGG_2021_Human', 'Reactome_Pathways_2024'])

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
    depmap_file = ctx.paths['data_dir'] / 'depmap' / 'CRISPRGeneEffect.csv'
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
    try:
        test_loader = TrainingLoader(batch_size=ctx.config['batch_size'], split='test', data_dir=ctx.paths['train_dir'], device=ctx.device, return_cell_type=True)
    except KeyError:
        return {'error': 'cell_type field not found in data shards'}

    test_steps = ctx.config['test_total_examples'] // ctx.config['batch_size']

    all_emb, all_cell_type = [], []
    with torch.no_grad():
        for _ in tqdm(range(test_steps), desc='cell_type_probing: Extracting embeddings'):
            batch = test_loader.next_batch()
            cont_x, cont_tot = batch[0], batch[1]
            cell_type = batch[-1]
            emb = ctx.biojepa.student(cont_x, cont_tot, mask_idx=None).mean(dim=1).cpu().numpy()
            all_emb.append(emb)
            all_cell_type.append(cell_type.cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0)
    cell_types = np.concatenate(all_cell_type, axis=0).flatten()

    unique_types = sorted(np.unique(cell_types))
    type_map = {t: i for i, t in enumerate(unique_types)}
    labels = np.array([type_map[t] for t in cell_types])
    n_classes = len(type_map)

    if n_classes < 2:
        return {'error': 'Only one cell type in data - eval requires multi-cell-type data', 'config': {'num_cell_types': n_classes}}

    train_idx, val_idx = train_test_split(np.arange(len(embeddings)), test_size=0.2, random_state=42, stratify=labels)

    print('Training cell type classifier...')
    _, val_preds, val_acc = train_linear_classifier(embeddings[train_idx], labels[train_idx], embeddings[val_idx], labels[val_idx], n_classes, ctx.device, epochs=100)

    macro_f1 = f1_score(labels[val_idx], val_preds, average='macro')
    chance = 1.0 / n_classes

    print(f'cell_type_probing: Accuracy={val_acc:.4f} ({val_acc/chance:.1f}x chance), Macro F1={macro_f1:.4f}')

    return {
        'config': {'samples': len(embeddings), 'embedding_dim': int(embeddings.shape[1]), 'num_cell_types': n_classes},
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
            cont_x, cont_tot, _, _, _, _, _ = test_loader.next_batch()
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
            cont_x, cont_tot, case_x, case_tot, _, _, _ = test_loader.next_batch()
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
            cont_x, cont_tot, case_x, case_tot, p_idx, _, _ = test_loader.next_batch()
            case_z = ctx.biojepa.student(case_x, case_tot, mask_idx=None).mean(dim=1).cpu().numpy()
            all_emb.append(case_z)
            all_pert.append(p_idx.cpu().numpy())

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
            cont_x, cont_tot, _, _, _, _, _ = test_loader.next_batch()
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
    pert_ids = inf['pert_ids']
    mean_pred_deltas, mean_real_deltas = inf['mean_pred_deltas'], inf['mean_real_deltas']
    mean_pred_abs, mean_real_abs = inf['mean_pred_abs'], inf['mean_real_abs']
    sample_mses, sample_correlations = inf['sample_mses'], inf['sample_correlations']
    n_genes = ctx.config['num_genes']

    TOP_K = 50
    per_pert_r2_all, per_pert_r2_top50, per_pert_mse = [], [], []

    for pid in pert_ids:
        pred_abs, real_abs = mean_pred_abs[pid], mean_real_abs[pid]
        pred_delta, real_delta = mean_pred_deltas[pid], mean_real_deltas[pid]

        if np.std(real_abs) > 1e-9:
            per_pert_r2_all.append(r2_score(real_abs, pred_abs))
        top_k_idx = np.argsort(np.abs(real_delta))[-TOP_K:]
        per_pert_r2_top50.append(r2_score(real_abs[top_k_idx], pred_abs[top_k_idx]))
        per_pert_mse.append(np.mean((pred_delta - real_delta)**2))

    per_pert_r2_all, per_pert_r2_top50 = np.array(per_pert_r2_all), np.array(per_pert_r2_top50)
    per_pert_mse = np.array(per_pert_mse)

    pred_severity = np.array([np.linalg.norm(mean_pred_deltas[pid]) for pid in pert_ids])
    real_severity = np.array([np.linalg.norm(mean_real_deltas[pid]) for pid in pert_ids])
    severity_pearson, _ = pearsonr(pred_severity, real_severity)
    severity_spearman, _ = spearmanr(pred_severity, real_severity)

    all_pred = np.concatenate([mean_pred_deltas[pid] for pid in pert_ids])
    all_real = np.concatenate([mean_real_deltas[pid] for pid in pert_ids])
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
        'config': {'test_perturbations': len(pert_ids), 'genes': n_genes, 'test_samples': len(sample_mses)},
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
    pert_ids = inf['pert_ids']
    mean_pred_deltas, mean_real_deltas = inf['mean_pred_deltas'], inf['mean_real_deltas']
    n_genes = ctx.config['num_genes']

    def classify_direction(delta, threshold=direction_threshold):
        direction = np.zeros_like(delta, dtype=np.int8)
        direction[delta >= threshold] = 1
        direction[delta <= -threshold] = -1
        return direction

    all_pred_dir = np.concatenate([classify_direction(mean_pred_deltas[pid]) for pid in pert_ids])
    all_real_dir = np.concatenate([classify_direction(mean_real_deltas[pid]) for pid in pert_ids])
    overall_accuracy = accuracy_score(all_real_dir, all_pred_dir)
    f1_up = f1_score(all_real_dir, all_pred_dir, labels=[1], average='macro', zero_division=0)
    f1_down = f1_score(all_real_dir, all_pred_dir, labels=[-1], average='macro', zero_division=0)
    f1_unchanged = f1_score(all_real_dir, all_pred_dir, labels=[0], average='macro', zero_division=0)

    TOP_K_DIR = 50
    top_deg_pred, top_deg_real = [], []
    for pid in pert_ids:
        top_k_idx = np.argsort(np.abs(mean_real_deltas[pid]))[-TOP_K_DIR:]
        top_deg_pred.append(classify_direction(mean_pred_deltas[pid][top_k_idx]))
        top_deg_real.append(classify_direction(mean_real_deltas[pid][top_k_idx]))
    top_deg_accuracy = accuracy_score(np.concatenate(top_deg_real), np.concatenate(top_deg_pred))

    all_magnitudes, all_correct = [], []
    for pid in pert_ids:
        real_delta, pred_delta = mean_real_deltas[pid], mean_pred_deltas[pid]
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
    for pid in pert_ids:
        pred_rank = np.argsort(np.abs(mean_pred_deltas[pid]))[::-1]
        true_rank = np.argsort(np.abs(mean_real_deltas[pid]))[::-1]
        for k in K_VALUES:
            deg_results[k]['precision'].append(precision_at_k(pred_rank, true_rank, k))
            deg_results[k]['ndcg'].append(ndcg_at_k(pred_rank, true_rank, k))
            deg_results[k]['overlap'].append(len(set(pred_rank[:k]) & set(true_rank[:k])))

    print(f'gene_level_analysis: Dir_acc={overall_accuracy:.4f}, Top50_acc={top_deg_accuracy:.4f}')

    return {
        'config': {'test_perturbations': len(pert_ids), 'genes': n_genes, 'direction_threshold': direction_threshold},
        'direction_of_effect': {'all_genes_accuracy': float(overall_accuracy), 'top50_degs_accuracy': float(top_deg_accuracy), 'f1_up': float(f1_up), 'f1_down': float(f1_down), 'f1_unchanged': float(f1_unchanged), 'accuracy_by_magnitude': accuracy_by_magnitude},
        'top_deg_recovery': {str(k): {'precision': float(np.mean(deg_results[k]['precision'])), 'ndcg': float(np.mean(deg_results[k]['ndcg'])), 'overlap': float(np.mean(deg_results[k]['overlap'])), 'vs_random': float(np.mean(deg_results[k]['overlap']) / (k * k / n_genes))} for k in K_VALUES}
    }


def _perturbation_retrieval(ctx, n_eval=100):
    '''Given desired outcome, can we find the right perturbation?'''
    inf = ctx.test_inference
    pert_ids = inf['pert_ids']
    mean_real_deltas = inf['mean_real_deltas']
    mean_control_states = inf['mean_control_states']
    n_genes, n_perturbations = ctx.config['num_genes'], ctx.input_bank.shape[0]

    pert_mod = torch.zeros(n_perturbations, dtype=torch.long, device=ctx.device)
    pert_mode = torch.zeros(n_perturbations, dtype=torch.long, device=ctx.device)

    def predict_all_deltas(control_x_np, batch_size=64):
        control_x = torch.from_numpy(control_x_np).float().to(ctx.device)
        control_tot = control_x.sum()
        all_pred = []
        for start in range(0, n_perturbations, batch_size):
            end = min(start + batch_size, n_perturbations)
            B = end - start
            control_batch = control_x.unsqueeze(0).expand(B, -1)
            control_tot_batch = control_tot.unsqueeze(0).expand(B)
            batch_idx = torch.arange(start, end, device=ctx.device)
            with torch.no_grad():
                z_ctx = ctx.biojepa.student(control_batch, control_tot_batch, mask_idx=None)
                action = ctx.biojepa.composer(ctx.input_bank[batch_idx], pert_mod[batch_idx], pert_mode[batch_idx])
                targets = torch.arange(n_genes, device=ctx.device).unsqueeze(0).expand(B, -1)
                z_pred, _ = ctx.biojepa.predictor(z_ctx, action, targets)
                all_pred.append((ctx.decoder(z_pred) - ctx.decoder(z_ctx)).cpu().numpy())
        return np.concatenate(all_pred, axis=0)

    def cos_sim(a, b):
        a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
        b_n = b / (np.linalg.norm(b) + 1e-8)
        return np.dot(a_n, b_n)

    eval_perts = pert_ids[:min(len(pert_ids), n_eval)]
    ranks = []
    for pid in tqdm(eval_perts, desc='perturbation_retrieval: Evaluating'):
        sims = cos_sim(predict_all_deltas(mean_control_states[pid]), mean_real_deltas[pid])
        ranks.append(np.where(np.argsort(sims)[::-1] == pid)[0][0] + 1)

    ranks = np.array(ranks)
    K_VALUES = [1, 5, 10, 20, 50]

    print(f'perturbation_retrieval: MRR={np.mean(1.0/ranks):.4f}, Median_rank={np.median(ranks):.0f}')

    return {
        'config': {'test_perturbations_evaluated': len(eval_perts), 'total_perturbations_in_bank': n_perturbations, 'genes': n_genes},
        'metrics': {'mrr': float(np.mean(1.0/ranks)), 'median_rank': float(np.median(ranks)), 'mean_rank': float(np.mean(ranks))},
        'recall_at_k': {str(k): {'recall': float(np.mean(ranks <= k)), 'vs_random': float(np.mean(ranks <= k) / (k / n_perturbations))} for k in K_VALUES}
    }


def _uncertainty_calibration(ctx, n_bins=10):
    '''Are confidence estimates meaningful?'''
    inf = ctx.test_inference
    pred_deltas = inf['sample_pred_deltas']
    real_deltas = inf['sample_real_deltas']
    sample_logvars = inf['sample_logvars']
    pert_ids = inf['sample_pert_ids']

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
    ece = sum((((unc_norm >= i/n_bins) & (unc_norm < (i+1)/n_bins if i < n_bins-1 else unc_norm <= 1)).sum() / len(sample_mse)) *
              abs(unc_norm[(unc_norm >= i/n_bins) & (unc_norm < (i+1)/n_bins if i < n_bins-1 else unc_norm <= 1)].mean() -
                  err_norm[(unc_norm >= i/n_bins) & (unc_norm < (i+1)/n_bins if i < n_bins-1 else unc_norm <= 1)].mean())
              for i in range(n_bins) if ((unc_norm >= i/n_bins) & (unc_norm < (i+1)/n_bins if i < n_bins-1 else unc_norm <= 1)).sum() > 0)

    pert_unc, pert_err = defaultdict(list), defaultdict(list)
    for i, pid in enumerate(pert_ids):
        pert_unc[pid].append(sample_unc[i])
        pert_err[pid].append(sample_mse[i])
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
    with open(ctx.paths['pert_dir'] / 'input_to_id.json') as f:
        input_to_id = json.load(f)
    id_to_gene = {pid: key.split('_')[0].upper() for key, pid in input_to_id.items()}

    pathway_libs = load_pathway_annotations(['KEGG_2021_Human'])
    n_perts = ctx.input_bank.shape[0]
    with torch.no_grad():
        action_vectors = ctx.biojepa.composer(ctx.input_bank, torch.zeros(n_perts, dtype=torch.long, device=ctx.device), torch.zeros(n_perts, dtype=torch.long, device=ctx.device)).cpu().numpy()

    gene_to_pathway = {}
    for pathway, genes in pathway_libs['KEGG_2021_Human'].items():
        if 15 <= len(genes) <= 300:
            for gene in genes:
                if gene.upper() not in gene_to_pathway:
                    gene_to_pathway[gene.upper()] = pathway

    pert_labels = {pid: gene_to_pathway[gene] for pid, gene in id_to_gene.items() if gene in gene_to_pathway}
    action_idx = list(pert_labels.keys())
    metrics = compute_pathway_clustering_metrics(action_vectors[action_idx], [pert_labels[i] for i in action_idx], min_samples_per_class=5)

    print(f'action_vector_pathways: KEGG sil={metrics["silhouette_score"]:.4f}, kNN={metrics["knn_accuracy"]:.4f}')
    return {'config': {'n_perturbations': len(id_to_gene)}, 'kegg': metrics}


def _moa_matching(ctx):
    '''Do same-pathway perturbations produce similar predicted effects?'''
    inf = ctx.test_inference
    pathway_libs = load_pathway_annotations(['KEGG_2021_Human'])

    with open(ctx.paths['pert_dir'] / 'input_to_id.json') as f:
        input_to_id = json.load(f)
    id_to_gene = {pid: key.split('_')[0].upper() for key, pid in input_to_id.items()}

    gene_to_pathway = {}
    for pathway, genes in pathway_libs['KEGG_2021_Human'].items():
        if 15 <= len(genes) <= 200:
            for gene in genes:
                if gene.upper() not in gene_to_pathway:
                    gene_to_pathway[gene.upper()] = pathway

    pert_to_pathway = {pid: gene_to_pathway[gene] for pid, gene in id_to_gene.items() if gene in gene_to_pathway}
    test_perts = set(inf['pert_ids'])
    valid_perts = [pid for pid in pert_to_pathway if pid in test_perts]

    pathway_to_perts = defaultdict(list)
    for pid in valid_perts:
        pathway_to_perts[pert_to_pathway[pid]].append(pid)
    valid_pathways = {p: perts for p, perts in pathway_to_perts.items() if len(perts) >= 3}

    if len(valid_pathways) < 2:
        return {'error': 'Not enough valid pathways'}

    all_perts = [pid for perts in valid_pathways.values() for pid in perts]
    pert_to_idx = {pid: i for i, pid in enumerate(all_perts)}
    delta_matrix = np.array([inf['mean_pred_deltas'][pid] for pid in all_perts])
    sim_matrix = cosine_similarity(delta_matrix)

    within_sims, between_sims = [], []
    for pathway, perts in valid_pathways.items():
        idx = [pert_to_idx[p] for p in perts]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                within_sims.append(sim_matrix[idx[i], idx[j]])

    pathways = list(valid_pathways.keys())
    for i, p1 in enumerate(pathways):
        for j in range(i + 1, len(pathways)):
            p2 = pathways[j]
            for pid1 in valid_pathways[p1]:
                for pid2 in valid_pathways[p2]:
                    between_sims.append(sim_matrix[pert_to_idx[pid1], pert_to_idx[pid2]])

    within_sims, between_sims = np.array(within_sims), np.array(between_sims)
    mean_within, mean_between = np.mean(within_sims), np.mean(between_sims)
    ratio = mean_within / mean_between if mean_between != 0 else float('inf')
    _, p_val = mannwhitneyu(within_sims, between_sims, alternative='greater')

    print(f'moa_matching: Within={mean_within:.4f}, Between={mean_between:.4f}, Ratio={ratio:.3f}x')

    return {
        'config': {'n_pathways': len(valid_pathways), 'n_perturbations': len(all_perts)},
        'similarity': {'mean_within_pathway': float(mean_within), 'mean_between_pathway': float(mean_between), 'similarity_ratio': float(ratio), 'mann_whitney_p': float(p_val), 'n_within_pairs': len(within_sims), 'n_between_pairs': len(between_sims)}
    }
