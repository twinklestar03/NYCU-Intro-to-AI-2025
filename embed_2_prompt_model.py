from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    GPT2LMHeadModel,
)

class Embed2Prompt(nn.Module):
    '''Two-layer projector with LayerNorm + GELU.'''

    def __init__(self, dim_in: int, dim_out: int, prefix_len: int = 10, dropout: float = 0.1):
        super().__init__()
        self.prefix_len = prefix_len
        self.dim_out = dim_out
        self.norm = nn.LayerNorm(dim_in)
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out * 4),  # widen factor 4
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_out * 4, dim_out * prefix_len),
            nn.Tanh(),                       # keep in (‑1,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D_in)
        x = self.norm(x.float())
        out = self.net(x)
        return out.view(x.size(0), self.prefix_len, self.dim_out)


class Embed2PromptModel(nn.Module):
    def __init__(self, base_lm: GPT2LMHeadModel, bridge: Embed2Prompt, tokenizer, freeze_lm: bool = True):
        super().__init__()
        self.lm = base_lm
        self.bridge = bridge
        self.tokenizer = tokenizer
        self.prefix_len = bridge.prefix_len
        if freeze_lm:
            self.lm.requires_grad_(False)

    def forward(self, emb_vecs: torch.Tensor, labels: torch.Tensor):
        prefix = self.bridge(emb_vecs)                             # (B,P,H)
        ids = labels.clone()
        ids[labels == -100] = self.tokenizer.pad_token_id          # ensure indexable
        tok_embeds = self.lm.transformer.wte(ids)                  # (B,L,H)
        inputs_embeds = torch.cat([prefix, tok_embeds], dim=1)
        pad = torch.full((labels.size(0), self.prefix_len), -100, device=labels.device)
        extended_labels = torch.cat([pad, labels], dim=1)
        return self.lm(inputs_embeds=inputs_embeds, labels=extended_labels)
