#!/usr/bin/env python
'''Run AC training evaluation suite.'''
import argparse
import os
from pathlib import Path
from .evals import EvalContext, run_ac_evals, save_report


def main():
    parser = argparse.ArgumentParser(description='Run AC training evaluation suite')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file (absolute, or relative to checkpoint dir)')
    parser.add_argument('--output', default='eval_report.json', help='Output file for results')
    parser.add_argument('--data-root', default=str(Path('~/data/jepa/v1_0').expanduser()), help='Data root directory')
    parser.add_argument('--checkpoint-root', default=str(Path('~/data/jepa/v1_0').expanduser()), help='Checkpoint root directory')
    parser.add_argument('--ref-dir', default=None, help='Reference data directory (default: $BIOJEPA_REF_DIR or <data_root>/../reference_data)')
    parser.add_argument('--num-genes', type=int, required=True, help='Number of genes in model')
    parser.add_argument('--embed-dim', type=int, required=True, help='Embedding dimension')
    parser.add_argument('--n-layer', type=int, required=True, help='Number of transformer layers')
    parser.add_argument('--n-pre-layer', type=int, default=2, help='Number of masked predictor layers')
    parser.add_argument('--heads', type=int, required=True, help='Number of attention heads')
    parser.add_argument('--pert-latent-dim', type=int, default=128, help='Composer latent dimension')
    parser.add_argument('--pert-mode-dim', type=int, default=64, help='Composer mode dimension')
    parser.add_argument('--predictor-embed-dim', type=int, default=128, help='Predictor embedding dimension')
    parser.add_argument('--predictor-n-layer', type=int, default=4, help='Number of predictor layers')
    parser.add_argument('--predictor-heads', type=int, default=4, help='Number of predictor attention heads')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    args = parser.parse_args()
    data_root = Path(args.data_root)
    checkpoint_root = Path(args.checkpoint_root)
    ref_dir = Path(args.ref_dir) if args.ref_dir else Path(os.environ.get('BIOJEPA_REF_DIR', data_root.parent / 'reference_data'))
    if not ref_dir.exists():
        raise FileNotFoundError(
            f'Reference directory not found: {ref_dir}. '
            f'Expected a directory containing resources such as depmap/ and gene_family/.'
        )

    config = {
        'num_genes': args.num_genes,
        'embed_dim': args.embed_dim,
        'n_layer': args.n_layer,
        'n_pre_layer': args.n_pre_layer,
        'heads': args.heads,
        'pert_latent_dim': args.pert_latent_dim,
        'pert_mode_dim': args.pert_mode_dim,
        'predictor_embed_dim': args.predictor_embed_dim,
        'predictor_n_layer': args.predictor_n_layer,
        'predictor_heads': args.predictor_heads,
        'batch_size': args.batch_size,
        'checkpoint_path': args.checkpoint,
    }

    print('=' * 60)
    print('AC Training Evaluation Suite')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Reference directory: {ref_dir}')
    print('=' * 60)

    ctx = EvalContext(config=config, data_root=data_root, checkpoint_root=checkpoint_root, ref_dir=ref_dir)
    results = run_ac_evals(ctx)
    save_report(results, args.output)

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  executed_evals: {", ".join(results.keys())}')
    for name, res in results.items():
        print(f'  {name}: {"ERROR" if "error" in res else "OK"}')


if __name__ == '__main__':
    main()
