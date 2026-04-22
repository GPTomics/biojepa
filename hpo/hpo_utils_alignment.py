'''HPO utility functions for BioJEPA v0.7 composer training optimization.'''

import numpy as np


def is_valid_alignment_config(params, heads=4):
    return params['pert_latent_dim'] % heads == 0


def compute_alignment_step_budget(batch_size, epochs=50, total_samples=10800):
    steps_per_epoch = total_samples // batch_size
    return epochs * steps_per_epoch


def compute_alignment_objective(eval_results):
    '''7-metric weighted composite. Returns float in [0, 1].

    v3 scaling calibrated from v0.6 InfoNCE observed ranges (46 completed trials):
      dna_cosine_sim: 0.25-0.81, dna_mrr: 0.012-0.032, consistency_ratio: 1.14-3.58
      mode_above_chance: 3.6-6.95, tfp_above_chance: 20-34, recovery_avg: 1.4-5.1
    Each metric scaled so median maps to ~0.5 and best maps to ~0.8-0.9 (room to improve).
    '''
    dna_mrr = eval_results.get('seq_to_target_retrieval', {}).get('by_modality', {}).get('dna', {}).get('mrr', 0)
    mrr_score = max(0, min(1, (dna_mrr - 0.01) / 0.03))

    dna_sim = eval_results.get('paired_alignment_quality', {}).get('by_modality', {}).get('dna', {}).get('mean_cosine_sim', 0)
    sim_score = max(0, min(1, (dna_sim - 0.3) / 0.6))

    consistency_ratio = eval_results.get('cross_modality_target_consistency', {}).get('metrics', {}).get('consistency_ratio', 1.0)
    consistency_score = max(0, min(1, (consistency_ratio - 1) / 0.5))

    mode_ratio = eval_results.get('mode_sensitivity', {}).get('classification', {}).get('above_chance_ratio', 1.0)
    mode_score = max(0, min(1, (mode_ratio - 1) / 10))

    semantic_gap = eval_results.get('mode_semantic_consistency', {}).get('embedding_semantics', {}).get('semantic_gap', 0.0)
    mode_semantic_score = max(0, min(1, semantic_gap / 0.3))

    tfp = eval_results.get('target_family_probing', {}).get('by_embedding_type', {})
    tfp_ratios = [tfp.get(k, {}).get('above_chance_ratio') for k in ['fused', 'seq_only']]
    tfp_ratios = [r for r in tfp_ratios if r is not None]
    tfp_ratio = np.mean(tfp_ratios) if tfp_ratios else 1.0
    tfp_score = max(0, min(1, (tfp_ratio - 1) / 40))

    robustness = eval_results.get('missing_data_robustness', {}).get('recovery_ratio', {})
    recovery_vals = [v for v in robustness.values() if isinstance(v, (int, float))]
    recovery_avg = np.mean(recovery_vals) if recovery_vals else 0
    robustness_score = max(0, min(1, recovery_avg / 5))

    weights = {'sim': 0.30, 'mrr': 0.20, 'consistency': 0.20, 'mode': 0.10, 'mode_semantic': 0.05, 'tfp': 0.10, 'robustness': 0.05}
    scores = {'sim': sim_score, 'mrr': mrr_score, 'consistency': consistency_score, 'mode': mode_score, 'mode_semantic': mode_semantic_score, 'tfp': tfp_score, 'robustness': robustness_score}
    return float(sum(weights[k] * scores[k] for k in weights))


def summarize_alignment_results(eval_results):
    '''Compact dict for trial user_attrs. Captures all metrics used in objective.'''
    summary = {}

    retrieval = eval_results.get('seq_to_target_retrieval', {}).get('by_modality', {})
    if 'dna' in retrieval:
        summary['dna_mrr'] = retrieval['dna'].get('mrr')
        summary['dna_recall_at_10'] = retrieval['dna'].get('recall_at_k', {}).get('10')
    if 'chemical' in retrieval:
        summary['chem_mrr'] = retrieval['chemical'].get('mrr')

    paired = eval_results.get('paired_alignment_quality', {}).get('by_modality', {})
    if 'dna' in paired:
        summary['dna_cosine_sim'] = paired['dna'].get('mean_cosine_sim')

    consistency = eval_results.get('cross_modality_target_consistency', {}).get('metrics', {})
    if 'consistency_ratio' in consistency:
        summary['consistency_ratio'] = consistency['consistency_ratio']

    mode = eval_results.get('mode_sensitivity', {}).get('classification', {})
    if 'above_chance_ratio' in mode:
        summary['mode_above_chance'] = mode['above_chance_ratio']

    mode_sem = eval_results.get('mode_semantic_consistency', {}).get('embedding_semantics', {})
    if 'semantic_gap' in mode_sem:
        summary['mode_semantic_gap'] = mode_sem['semantic_gap']
    cross_mode = eval_results.get('mode_semantic_consistency', {}).get('cross_mode_retrieval', {})
    if 'cross_mode_mrr' in cross_mode:
        summary['cross_mode_mrr'] = cross_mode['cross_mode_mrr']

    tfp = eval_results.get('target_family_probing', {}).get('by_embedding_type', {})
    tfp_ratios = [tfp.get(k, {}).get('above_chance_ratio') for k in ['fused', 'seq_only']]
    tfp_ratios = [r for r in tfp_ratios if r is not None]
    if tfp_ratios:
        summary['tfp_above_chance'] = float(np.mean(tfp_ratios))

    robustness = eval_results.get('missing_data_robustness', {}).get('recovery_ratio', {})
    recovery_vals = [v for v in robustness.values() if isinstance(v, (int, float))]
    if recovery_vals:
        summary['recovery_avg'] = float(np.mean(recovery_vals))

    return {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items() if v is not None}


def run_selected_alignment_evals(ctx, eval_names):
    '''Run only the specified alignment evals.'''
    from evals.evals import (
        _seq_to_target_retrieval, _cross_modality_target_consistency,
        _seq_target_gap_analysis, _paired_alignment_quality,
        _mode_sensitivity, _mode_semantic_consistency, _fusion_quality,
        _missing_data_robustness, _multi_pert_alignment, _target_family_probing,
        _cross_modality_alignment,
    )

    EVAL_FUNCTIONS = {
        'seq_to_target_retrieval': _seq_to_target_retrieval,
        'cross_modality_target_consistency': _cross_modality_target_consistency,
        'seq_target_gap_analysis': _seq_target_gap_analysis,
        'paired_alignment_quality': _paired_alignment_quality,
        'mode_sensitivity': _mode_sensitivity,
        'mode_semantic_consistency': _mode_semantic_consistency,
        'fusion_quality': _fusion_quality,
        'missing_data_robustness': _missing_data_robustness,
        'multi_pert_alignment': _multi_pert_alignment,
        'target_family_probing': _target_family_probing,
        'cross_modality_alignment': _cross_modality_alignment,
    }

    results = {}
    for name in eval_names:
        if name in EVAL_FUNCTIONS:
            results[name] = EVAL_FUNCTIONS[name](ctx)
    return results
