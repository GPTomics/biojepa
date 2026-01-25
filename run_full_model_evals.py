#!/usr/bin/env python
'''Run full model evaluations for BioJEPA v0.6.

Evaluations:
- expression_prediction: Can we predict gene expression after perturbation?
- gene_level_analysis: Direction of effect + top DEG recovery
- perturbation_retrieval: Given desired outcome, find the perturbation
- uncertainty_calibration: Are confidence estimates meaningful?
- action_vector_pathways: Do same-pathway perturbations have similar action vectors?
- moa_matching: Do same-pathway perturbations produce similar predicted effects?
'''
import argparse
from evals import EvalContext, run_full_model_evals, save_report

parser = argparse.ArgumentParser(description='Run full model evaluations for BioJEPA v0.6')
parser.add_argument('--output', default='eval_report.json', help='Output file for results')
parser.add_argument('--data-root', default='/Users/djemec/data/jepa/v0_6', help='Data root directory')
parser.add_argument('--checkpoint-root', default='/Users/djemec/data/jepa/v0_6', help='Checkpoint root directory')
args = parser.parse_args()

print('=' * 60)
print('BioJEPA v0.6 - Full Model Evaluations')
print('=' * 60)

ctx = EvalContext(data_root=args.data_root, checkpoint_root=args.checkpoint_root)
results = run_full_model_evals(ctx)
save_report(results, args.output)

print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
for name, res in results.items():
    print(f'  {name}: {"ERROR" if "error" in res else "OK"}')
