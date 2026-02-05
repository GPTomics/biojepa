#!/usr/bin/env python
'''Run pretraining evaluations for BioJEPA v0.6.

Evaluations:
- batch_invariance: Are representations confounded by batch effects?
- gene_embedding_pathways: Do genes in same pathway cluster together?
- essential_gene_prediction: Do gene embeddings encode functional importance?
'''
import argparse
from .evals import EvalContext, run_pretraining_evals, save_report


def main():
    parser = argparse.ArgumentParser(description='Run pretraining evaluations for BioJEPA v0.6')
    parser.add_argument('--output', default='eval_report.json', help='Output file for results')
    parser.add_argument('--data-root', default='/Users/djemec/data/jepa/v0_6', help='Data root directory')
    parser.add_argument('--checkpoint-root', default='/Users/djemec/data/jepa/v0_6', help='Checkpoint root directory')
    parser.add_argument('--num-genes', type=int, required=True, help='Number of genes in model')
    parser.add_argument('--embed-dim', type=int, required=True, help='Embedding dimension')
    parser.add_argument('--n-layer', type=int, required=True, help='Number of transformer layers')
    parser.add_argument('--heads', type=int, required=True, help='Number of attention heads')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    args = parser.parse_args()

    config = {
        'num_genes': args.num_genes,
        'embed_dim': args.embed_dim,
        'n_layer': args.n_layer,
        'heads': args.heads,
        'batch_size': args.batch_size
    }

    print('=' * 60)
    print('BioJEPA v0.6 - Pretraining Evaluations')
    print('=' * 60)

    ctx = EvalContext(config=config, data_root=args.data_root, checkpoint_root=args.checkpoint_root)
    results = run_pretraining_evals(ctx)
    save_report(results, args.output)

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for name, res in results.items():
        print(f'  {name}: {"ERROR" if "error" in res else "OK"}')


if __name__ == '__main__':
    main()
