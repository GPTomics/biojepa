#!/usr/bin/env python
'''Run pretraining evaluation suite.'''
import argparse
import os
from pathlib import Path
from .evals import EvalContext, run_pretraining_evals, save_report


def main():
    parser = argparse.ArgumentParser(description='Run pretraining evaluation suite')
    parser.add_argument('--output', default='eval_report.json', help='Output file for results')
    parser.add_argument('--data-root', default='/Users/djemec/data/jepa/v0_6', help='Data root directory')
    parser.add_argument('--checkpoint-root', default='/Users/djemec/data/jepa/v0_6', help='Checkpoint root directory')
    parser.add_argument('--ref-dir', default=None, help='Reference data directory (default: $BIOJEPA_REF_DIR or <data_root>/references)')
    parser.add_argument('--num-genes', type=int, required=True, help='Number of genes in model')
    parser.add_argument('--embed-dim', type=int, required=True, help='Embedding dimension')
    parser.add_argument('--n-layer', type=int, required=True, help='Number of transformer layers')
    parser.add_argument('--heads', type=int, required=True, help='Number of attention heads')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    args = parser.parse_args()
    data_root = Path(args.data_root)
    checkpoint_root = Path(args.checkpoint_root)
    ref_dir = Path(args.ref_dir) if args.ref_dir else Path(os.environ.get('BIOJEPA_REF_DIR', data_root / 'references'))
    if not ref_dir.exists():
        raise FileNotFoundError(
            f'Reference directory not found: {ref_dir}. '
            f'Expected a directory containing resources such as depmap/ and gene_family/.'
        )

    config = {
        'num_genes': args.num_genes,
        'embed_dim': args.embed_dim,
        'n_layer': args.n_layer,
        'heads': args.heads,
        'batch_size': args.batch_size
    }

    print('=' * 60)
    print('Pretraining Evaluation Suite')
    print(f'Reference directory: {ref_dir}')
    print('=' * 60)

    ctx = EvalContext(config=config, data_root=data_root, checkpoint_root=checkpoint_root, ref_dir=ref_dir)
    results = run_pretraining_evals(ctx)
    save_report(results, args.output)

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  executed_evals: {", ".join(results.keys())}')
    for name, res in results.items():
        print(f'  {name}: {"ERROR" if "error" in res else "OK"}')


if __name__ == '__main__':
    main()
