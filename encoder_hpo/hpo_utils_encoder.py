'''HPO utility functions for BioJEPA v0.7 encoder training optimization.

Designed for short-horizon (4-epoch) search with full 50-epoch schedule. See plan
at ~/.claude/plans/serialized-stirring-squid.md for design rationale.
'''

import gc
import math
import random
import traceback

import numpy as np
import torch


# Reference values for normalization. (random_baseline, v0.6_final).
# v0.6 final is the ep50 deployed checkpoint.
NORMALIZATION_TABLE = {
    'essential_auroc':    (0.50, 0.71),
    'recon_r_squared':    (0.00, 0.98),
    'cell_type_macro_f1': (0.25, 0.70),
}

GATE_METRICS = ['essential_auroc', 'recon_r_squared', 'cell_type_macro_f1']

# Reference scores for context in deepdive analysis
V06_REFERENCE = {
    'essential_auroc': 0.7049, 'recon_r_squared': 0.977,
    'cell_type_macro_f1': 0.701, 'composite': 1.0,
}
V07_OLD_EP42_REFERENCE = {
    'essential_auroc': 0.620, 'recon_r_squared': 0.910,
    'cell_type_macro_f1': 0.566, 'composite': 0.734,
}
V07_NEW_EP2_REFERENCE = {
    'essential_auroc': 0.649, 'recon_r_squared': 0.933,
    'cell_type_macro_f1': 0.528, 'composite': 0.760,
}


def normalize_metric(name, value):
    '''Map raw metric value to fraction-of-v0.6 score in [0, 1]+ range.'''
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    rand, v06 = NORMALIZATION_TABLE[name]
    return (value - rand) / (v06 - rand)


def composite(metrics_dict):
    '''Equal-weighted composite of normalized gate metrics. Returns float.

    v0.6 ep50 = 1.0 by construction.
    Random encoder = 0.0.
    '''
    norms = [normalize_metric(k, metrics_dict.get(k)) for k in GATE_METRICS]
    return sum(norms) / len(GATE_METRICS)


def derive_vicreg_coeffs(sim_coeff, std_to_sim_ratio, cov_to_sim_ratio):
    '''Convert search-space ratios to absolute VICReg coefficients.'''
    return {
        'sim_coeff': sim_coeff,
        'std_coeff': sim_coeff * std_to_sim_ratio,
        'cov_coeff': sim_coeff * cov_to_sim_ratio,
    }


# Pruning thresholds — see plan section "Pruning rules"
HARD_FLOORS = {
    1: ('essential_auroc', 0.45),
    2: ('essential_auroc', 0.52),
}
COMPOSITE_HARD_FLOOR_EP3 = 0.40
OUTLIER_SIGMA = 1.5
OUTLIER_MIN_TRIALS = 5
INTRA_TRIAL_COLLAPSE_THRESHOLDS = {
    'essential_auroc': 0.05,
    'recon_r_squared': 0.10,
    'cell_type_macro_f1': 0.10,
}


def _hard_floor_check(epoch, metrics):
    '''Layer 1: hard absolute thresholds for catastrophically broken trials.'''
    if epoch in HARD_FLOORS:
        metric_name, floor = HARD_FLOORS[epoch]
        if metrics.get(metric_name) is not None and metrics[metric_name] < floor:
            return f'hard_floor:{metric_name}_at_ep{epoch}'
    if epoch == 3 and composite(metrics) < COMPOSITE_HARD_FLOOR_EP3:
        return f'hard_floor:composite_ep3'
    return None


def _outlier_check(epoch, metrics, completed_metrics_at_epoch):
    '''Layer 2: per-metric population outlier (sigma=1.5) vs other completed trials.

    completed_metrics_at_epoch: list of metric dicts from completed trials at this epoch.
    '''
    if epoch < 2 or len(completed_metrics_at_epoch) < OUTLIER_MIN_TRIALS:
        return None
    for m in GATE_METRICS:
        values = [d.get(m) for d in completed_metrics_at_epoch if d.get(m) is not None]
        if len(values) < OUTLIER_MIN_TRIALS:
            continue
        mean_v, std_v = float(np.mean(values)), float(np.std(values))
        if metrics.get(m) is not None and metrics[m] < mean_v - OUTLIER_SIGMA * std_v:
            return f'outlier:{m}_at_ep{epoch}_mean{mean_v:.3f}_std{std_v:.3f}'
    return None


def _intra_trial_collapse_check(epoch, metrics, prev_metrics):
    '''Layer 3: epoch-over-epoch metric drop (mirrors recon-cliff observed in current run).'''
    if epoch < 2 or prev_metrics is None:
        return None
    for m, thresh in INTRA_TRIAL_COLLAPSE_THRESHOLDS.items():
        prev_v = prev_metrics.get(m)
        curr_v = metrics.get(m)
        if prev_v is None or curr_v is None:
            continue
        if prev_v - curr_v > thresh:
            return f'collapse:{m}_drop{prev_v - curr_v:.3f}_at_ep{epoch}'
    return None


def should_prune(epoch, metrics, prev_metrics, completed_metrics_at_epoch):
    '''Apply three custom pruning layers. Returns reason string if should prune, None otherwise.

    Args:
        epoch: 1-indexed eval epoch number.
        metrics: dict from summarize_encoder_evals at this epoch.
        prev_metrics: dict from previous eval (epoch-1), or None for ep1.
        completed_metrics_at_epoch: list of metric dicts from prior completed trials at same epoch.
    '''
    reason = _hard_floor_check(epoch, metrics)
    if reason: return reason
    reason = _outlier_check(epoch, metrics, completed_metrics_at_epoch)
    if reason: return reason
    reason = _intra_trial_collapse_check(epoch, metrics, prev_metrics)
    if reason: return reason
    return None


def collect_completed_metrics_at_epoch(study, epoch):
    '''Pull eval_epoch_{epoch} metrics from all COMPLETE/PRUNED trials in study (skip current trial).

    Used for outlier-vs-population pruning.
    '''
    import optuna
    out = []
    for t in study.trials:
        if t.state not in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED):
            continue
        m = t.user_attrs.get(f'eval_epoch_{epoch}')
        if m is not None:
            out.append(m)
    return out


def reset_all_seeds(seed=1337):
    '''Reset all four RNG sources. Must be called BEFORE create_model.

    See plan section "Seed handling" for rationale.
    '''
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision('high')


def run_encoder_eval_for_hpo(model, model_cfg, batch_size, data_root, ref_dir):
    '''Run all encoder evals via EvalContext + biojepa injection.

    Mirrors training_v0_7.py:286-315 mid-training-eval pattern. Returns
    summarize_encoder_evals dict (compact metrics).

    Caller is responsible for setting model.eval() before and model.train() after.
    '''
    from evals.evals import EvalContext, run_encoder_evals, summarize_encoder_evals
    eval_config = {
        'num_genes': model_cfg.num_genes, 'embed_dim': model_cfg.embed_dim,
        'n_layer': model_cfg.n_layer, 'heads': model_cfg.heads,
        'batch_size': batch_size, 'verbose': False, 'eval_split': 'val',
    }
    eval_ctx = EvalContext(
        config=eval_config, data_root=data_root,
        checkpoint_root=data_root, ref_dir=ref_dir,
    )
    eval_ctx._biojepa = model
    try:
        raw = run_encoder_evals(eval_ctx)
        metrics = summarize_encoder_evals(raw)
    finally:
        eval_ctx._biojepa = None
        del eval_ctx
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return metrics


def fail_trial_with_traceback(trial, exc):
    '''Log exception details to trial user_attrs for post-hoc deepdive analysis.'''
    trial.set_user_attr('fail_reason', str(exc))
    trial.set_user_attr('traceback', traceback.format_exc())
