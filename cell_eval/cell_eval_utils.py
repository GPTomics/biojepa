import json
import gc
import torch
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_pert_name_mappings(pert_dir):
    pert_dir = Path(pert_dir)
    seq_dir = pert_dir / 'seq_banks'
    tgt_dir = pert_dir / 'target_banks'

    dna_idx_to_gene = {}
    dna_path = seq_dir / 'dna_to_idx.json'
    if dna_path.exists():
        with open(dna_path) as f:
            dna_to_idx = json.load(f)
        dna_idx_to_gene = {idx: key.split('_')[0].upper() for key, idx in dna_to_idx.items()}

    chem_idx_to_name = {}
    chem_path = seq_dir / 'chemical_to_idx.json'
    if chem_path.exists():
        with open(chem_path) as f:
            chem_to_idx = json.load(f)
        for key, idx in chem_to_idx.items():
            if idx not in chem_idx_to_name or (not key.startswith('C') and len(key) > 5):
                chem_idx_to_name[idx] = key

    target_idx_to_gene = {}
    tgt_path = tgt_dir / 'gene_to_target_idx.json'
    if tgt_path.exists():
        with open(tgt_path) as f:
            gene_to_target = json.load(f)
        target_idx_to_gene = {idx: gene for gene, idx in gene_to_target.items() if not gene.startswith('ENSG')}

    return dna_idx_to_gene, chem_idx_to_name, target_idx_to_gene


def get_pert_name(seq_idx, target_idx, modality, mode, n_perts, dna_map, chem_map, target_map, all_seq_idx=None, all_target_idx=None, all_modality=None, dose_raw=None):
    if n_perts > 1 and all_seq_idx is not None:
        names = []
        for j in range(n_perts):
            s, m = int(all_seq_idx[j]), int(all_modality[j])
            t = int(all_target_idx[j]) if all_target_idx is not None else -1
            if m == 2:
                names.append(chem_map.get(s, f'chem_{s}'))
            elif m == 0:
                names.append(dna_map.get(s, f'dna_{s}'))
            else:
                names.append(target_map.get(t, f'target_{t}'))
        name = '+'.join(sorted(names))
    elif int(modality) == 2:
        name = chem_map.get(int(seq_idx), f'chem_{seq_idx}')
    elif int(modality) == 0:
        name = dna_map.get(int(seq_idx), f'dna_{seq_idx}')
    else:
        name = target_map.get(int(target_idx), f'target_{target_idx}')

    if dose_raw is not None and int(modality) == 2 and float(dose_raw) > 0:
        name = f'{name}_{int(dose_raw)}nM'

    return name


def load_ntc_controls(data_root, dataset_name):
    path = Path(data_root) / 'ntc_controls' / f'{dataset_name}_ntc.npz'
    with np.load(path) as data:
        return {
            'expression': data['expression'].astype(np.float32),
            'total_counts': data['total_counts'].astype(np.float32),
            'gene_mask': data['gene_mask'].astype(np.bool_),
            'batch_ids': data['batch_ids'].astype(np.int64),
        }


def _make_pert_key(seq_idx, target_idx, modality, mode, n_perts, dose_raw):
    np_val = int(n_perts)
    return (
        int(seq_idx[0]), int(target_idx[0]),
        int(modality[0]), int(mode[0]), np_val,
        tuple(int(x) for x in seq_idx[:np_val]),
        tuple(int(x) for x in target_idx[:np_val]),
        tuple(int(x) for x in modality[:np_val]),
        int(dose_raw[0]) if modality[0] == 2 and dose_raw[0] > 0 else -1,
    )


def _scan_pert_metadata(shards):
    pert_meta = {}
    pert_counts = defaultdict(int)

    for shard_path in shards:
        with np.load(shard_path) as data:
            seq_idx = data['seq_idx'].astype(np.int64)
            target_idx = data['target_idx'].astype(np.int64)
            modality = data['modality'].astype(np.int64)
            mode = data['mode'].astype(np.int64)
            n_perts = data['n_perts'].astype(np.int64)
            dose_raw = data['dose'].astype(np.float32)
            has_seq = data['has_seq'].astype(np.bool_)
            has_target = data['has_target'].astype(np.bool_)

        for i in range(len(seq_idx)):
            key = _make_pert_key(seq_idx[i], target_idx[i], modality[i], mode[i], n_perts[i], dose_raw[i])
            pert_counts[key] += 1
            if key not in pert_meta:
                pert_meta[key] = {
                    'seq_idx': seq_idx[i].copy(), 'target_idx': target_idx[i].copy(),
                    'modality': modality[i].copy(), 'mode': mode[i].copy(),
                    'has_seq': has_seq[i].copy(), 'has_target': has_target[i].copy(),
                    'n_perts': int(n_perts[i]), 'dose_raw': dose_raw[i].copy(),
                }

    return pert_meta, pert_counts


def _collect_real_expression(shards, pert_key_to_name, max_samples, n_genes):
    pert_counts = defaultdict(int)
    total_available = defaultdict(int)
    for shard_path in shards:
        with np.load(shard_path) as data:
            seq_idx = data['seq_idx'].astype(np.int64)
            target_idx = data['target_idx'].astype(np.int64)
            modality = data['modality'].astype(np.int64)
            mode = data['mode'].astype(np.int64)
            n_perts_arr = data['n_perts'].astype(np.int64)
            dose_raw = data['dose'].astype(np.float32)
        for i in range(len(seq_idx)):
            key = _make_pert_key(seq_idx[i], target_idx[i], modality[i], mode[i], n_perts_arr[i], dose_raw[i])
            total_available[key] += 1

    n_perts_total = len(pert_key_to_name)
    total_samples = sum(total_available.values())
    if max_samples is not None and total_samples > max_samples:
        scale = max_samples / total_samples
        per_pert_cap = {k: max(1, int(v * scale)) for k, v in total_available.items()}
    else:
        per_pert_cap = {k: v for k, v in total_available.items()}

    real_chunks = []
    real_names = []
    chunk_samples = 0

    for shard_path in shards:
        with np.load(shard_path) as data:
            case = data['case'].astype(np.float32)
            seq_idx = data['seq_idx'].astype(np.int64)
            target_idx = data['target_idx'].astype(np.int64)
            modality = data['modality'].astype(np.int64)
            mode = data['mode'].astype(np.int64)
            n_perts_arr = data['n_perts'].astype(np.int64)
            dose_raw = data['dose'].astype(np.float32)

        shard_keep_mask = np.zeros(len(case), dtype=bool)
        shard_names = [''] * len(case)

        for i in range(len(case)):
            key = _make_pert_key(seq_idx[i], target_idx[i], modality[i], mode[i], n_perts_arr[i], dose_raw[i])
            if key not in pert_key_to_name:
                continue
            if pert_counts[key] >= per_pert_cap.get(key, 0):
                continue
            pert_counts[key] += 1
            shard_keep_mask[i] = True
            shard_names[i] = pert_key_to_name[key]

        kept = case[shard_keep_mask]
        if len(kept) > 0:
            real_chunks.append(kept)
            real_names.extend(n for n, k in zip(shard_names, shard_keep_mask) if k)
            chunk_samples += len(kept)

        del case

    if real_chunks:
        real_matrix = np.concatenate(real_chunks, axis=0)
    else:
        real_matrix = np.empty((0, n_genes), dtype=np.float32)

    return real_matrix, real_names


def build_cell_eval_adata(dataset_name, biojepa, decoder, seq_banks, target_bank, gene_names,
                          data_root, shard_dir, dna_map, chem_map, target_map,
                          get_seq_embeddings_fn, get_target_embeddings_fn,
                          n_ntc=64, max_samples_per_pert=None, max_samples=50000,
                          device='cuda', batch_size=64, verbose=True):
    ntc = load_ntc_controls(data_root, dataset_name)
    ntc_expr = ntc['expression'][:n_ntc]
    ntc_total = ntc['total_counts'][:n_ntc]
    gene_mask = ntc['gene_mask']
    n_ntc_actual = len(ntc_expr)
    N = len(gene_names)

    ntc_expr_t = torch.from_numpy(ntc_expr).to(device)
    ntc_total_t = torch.from_numpy(ntc_total).to(device)
    unknown_mask_t = torch.from_numpy(~gene_mask).unsqueeze(0).expand(n_ntc_actual, -1).to(device)

    test_dir = Path(shard_dir) / 'test'
    shards = sorted(test_dir.glob(f'shard_{dataset_name}_test_*.npz'))
    if not shards:
        raise FileNotFoundError(f'No test shards for {dataset_name} in {test_dir}')

    pert_meta, pert_sample_counts = _scan_pert_metadata(shards)

    pert_key_to_name = {}
    for key, meta in pert_meta.items():
        np_val = meta['n_perts']
        name = get_pert_name(
            meta['seq_idx'][0], meta['target_idx'][0],
            meta['modality'][0], meta['mode'][0], np_val,
            dna_map, chem_map, target_map,
            all_seq_idx=meta['seq_idx'][:np_val],
            all_target_idx=meta['target_idx'][:np_val],
            all_modality=meta['modality'][:np_val],
            dose_raw=meta['dose_raw'][0] if meta['modality'][0] == 2 else None,
        )
        pert_key_to_name[key] = name

    if verbose:
        total_real = sum(pert_sample_counts.values())
        print(f'{dataset_name}: {len(pert_meta)} perturbations, {total_real} real samples, {n_ntc_actual} NTC controls')

    all_pred_expr = []
    all_pred_names = []

    biojepa.eval()
    with torch.no_grad():
        z_context_ntc = biojepa.teacher(ntc_expr_t, ntc_total_t, mask_idx=None, unknown_mask=unknown_mask_t)

    for key, meta in tqdm(pert_meta.items(), desc=f'{dataset_name} perts', disable=not verbose):
        np_val = meta['n_perts']
        pert_name = pert_key_to_name[key]

        seq_idx_t = torch.from_numpy(meta['seq_idx']).unsqueeze(0).expand(n_ntc_actual, -1).to(device)
        target_idx_t = torch.from_numpy(meta['target_idx']).unsqueeze(0).expand(n_ntc_actual, -1).to(device)
        modality_t = torch.from_numpy(meta['modality']).unsqueeze(0).expand(n_ntc_actual, -1).to(device)
        mode_t = torch.from_numpy(meta['mode']).unsqueeze(0).expand(n_ntc_actual, -1).to(device)
        has_seq_t = torch.from_numpy(meta['has_seq']).unsqueeze(0).expand(n_ntc_actual, -1).to(device)
        has_target_t = torch.from_numpy(meta['has_target']).unsqueeze(0).expand(n_ntc_actual, -1).to(device)
        n_pert_slots = seq_idx_t.shape[1]
        pert_mask_t = torch.arange(n_pert_slots, device=device).unsqueeze(0) < np_val

        dose_val = meta['dose_raw'].copy()
        valid_dose = dose_val != -1.0
        dose_val = np.where(valid_dose, np.log1p(np.maximum(dose_val, 0.0)), dose_val)
        dose_t = torch.from_numpy(dose_val).unsqueeze(0).expand(n_ntc_actual, -1).to(device)

        seq_emb = get_seq_embeddings_fn(seq_idx_t, modality_t, seq_banks)
        target_emb = get_target_embeddings_fn(target_idx_t, target_bank)

        pred_batch = []
        with torch.no_grad():
            for start in range(0, n_ntc_actual, batch_size):
                end = min(start + batch_size, n_ntc_actual)
                sl = slice(start, end)
                b = end - start
                action_lat = biojepa.composer(
                    seq_emb[sl], target_emb[sl], modality_t[sl], mode_t[sl],
                    has_seq_t[sl], has_target_t[sl], pert_mask_t.expand(b, -1), dose=dose_t[sl]
                )
                tgt_idx = torch.arange(N, device=device).expand(b, N)
                z_pred_mu, _ = biojepa.predictor(z_context_ntc[sl], action_lat, tgt_idx)
                pred_delta = decoder(z_pred_mu) - decoder(z_context_ntc[sl])
                pred_expr = torch.clamp(ntc_expr_t[sl] + pred_delta, min=0.0)
                pred_np = pred_expr.cpu().numpy()
                pred_np[:, ~gene_mask] = 0.0
                pred_batch.append(pred_np)

        pred_matrix = np.concatenate(pred_batch, axis=0)
        all_pred_expr.append(pred_matrix)
        all_pred_names.extend([pert_name] * n_ntc_actual)

    pred_matrix = np.concatenate(all_pred_expr, axis=0) if all_pred_expr else np.empty((0, N), dtype=np.float32)
    del all_pred_expr
    gc.collect()

    if verbose:
        print(f'{dataset_name}: collecting real expression (streaming, max_samples={max_samples})...')
    real_matrix, all_real_names = _collect_real_expression(shards, pert_key_to_name, max_samples, N)

    ntc_matrix = ntc_expr[:n_ntc_actual]
    n_pred = pred_matrix.shape[0]
    n_real = real_matrix.shape[0]
    n_ctrl = ntc_matrix.shape[0]

    measured_idx = np.where(gene_mask)[0]
    pred_matrix = pred_matrix[:, measured_idx]
    real_matrix = real_matrix[:, measured_idx]
    ntc_matrix = ntc_matrix[:, measured_idx]
    measured_gene_names = [gene_names[i] for i in measured_idx]

    if verbose:
        print(f'{dataset_name}: adata_pred={n_pred + n_ctrl} rows, adata_real={n_real + n_ctrl} rows, {len(measured_idx)}/{N} genes measured')

    var_df = pd.DataFrame(index=measured_gene_names)

    obs_pred = pd.DataFrame({
        'perturbation': all_pred_names + ['control'] * n_ctrl,
    }, index=[f'pred_{i}' for i in range(n_pred)] + [f'ctrl_p_{i}' for i in range(n_ctrl)])
    adata_pred = ad.AnnData(X=np.vstack([pred_matrix, ntc_matrix]), obs=obs_pred, var=var_df)

    obs_real = pd.DataFrame({
        'perturbation': all_real_names + ['control'] * n_ctrl,
    }, index=[f'real_{i}' for i in range(n_real)] + [f'ctrl_r_{i}' for i in range(n_ctrl)])
    adata_real = ad.AnnData(X=np.vstack([real_matrix, ntc_matrix]), obs=obs_real, var=var_df)

    del pred_matrix, real_matrix
    gc.collect()

    return adata_pred, adata_real
