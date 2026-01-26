#!/usr/bin/env python
'''Run pretraining evaluations for BioJEPA v0.6.

Evaluations:
- batch_invariance: Are representations confounded by batch effects?
- gene_embedding_pathways: Do genes in same pathway cluster together?
- essential_gene_prediction: Do gene embeddings encode functional importance?
'''
import argparse
from evals.evals import EvalContext, run_pretraining_evals, save_report

parser = argparse.ArgumentParser(description='Run pretraining evaluations for BioJEPA v0.6')
parser.add_argument('--output', default='eval_report.json', help='Output file for results')
parser.add_argument('--data-root', default='/Users/djemec/data/jepa/v0_6', help='Data root directory')
parser.add_argument('--checkpoint-root', default='/Users/djemec/data/jepa/v0_6', help='Checkpoint root directory')
args = parser.parse_args()

print('=' * 60)
print('BioJEPA v0.6 - Pretraining Evaluations')
print('=' * 60)

ctx = EvalContext(data_root=args.data_root, checkpoint_root=args.checkpoint_root)
results = run_pretraining_evals(ctx)
save_report(results, args.output)

print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
for name, res in results.items():
    print(f'  {name}: {"ERROR" if "error" in res else "OK"}')
