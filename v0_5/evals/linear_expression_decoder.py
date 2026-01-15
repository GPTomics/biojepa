import torch
import torch.nn as nn
import math
from dataclasses import dataclass


def init_weights_robust(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        if isinstance(module, nn.Embedding):
            fan_in = module.embedding_dim
        else:
            fan_in = module.weight.size(1)
        std = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.02
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@dataclass
class BenchmarkDecoderConfig:
    embed_dim: int = 256


class BenchmarkDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head = nn.Linear(config.embed_dim, 1)
        self.apply(init_weights_robust)

    def forward(self, latents):
        gene_preds = self.head(latents)
        gene_preds = gene_preds.squeeze(-1)
        return gene_preds
