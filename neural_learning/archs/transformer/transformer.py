from typing import Union, Optional, Tuple
import logging

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class config():
    heads: int = 8
    layers: int = 8
    embeddings: int = 192
    xd : int = 2
    yd : int = 2
    variables: str = "v"
    type: str = "single"

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super.__init__()
        assert config.embeddings % config.heads == 0
        self.latt = nn.Linear(config.embeddings, 3*config.embeddings) 
        self.pr = nn.Linear(config.embeddings, config.embeddings)
        self.heads = config.heads
        self.embeddings = config.embeddings
        self.register_buffer('bias', torch.tril(torch.ones(config.nu, config.nu)).view(1, 1, config.nu, config.nu))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.latt(x)
        q, k, v = qkv.split(self.embeddings, dim=2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1,2)
        q = q.view(B, T, self.heads, C // self.heads).transpose(1,2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        x = att @ v
        x = self.pr(x)
        return x
    

class MLP(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.embeddings, 4*config.embeddings),
            nn.GELU(approximate='tanh'),
            nn.Linear(4*config.embeddings, config.embeddings)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class Block(nn.moduele):
    def __init__(self, config):
        super.__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(config.embeddings),
            CausalSelfAttention(config),
            nn.LayerNorm(config.embeddings),
            MLP(config)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class NeuronTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.neurontransformer = nn.ModuleDict(
            wte = nn.Embedding(),
            wpe = nn.Embedding(),
            h = nn.ModuleList(Block(config) for _ in range (config.layers)),
            lnf = nn.LayerNorm(config.embeddings) 
        )
        self.pr = nn.Linear()

        self._init_weigths()
        self._set_optimizers()

    @classmethod
    def import_weights(cls, weights: Optional[torch.Tensor]):
        pass
    
    @torch.no_grad()
    def _init_weigths(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.0)      # 0.0 => skip connection to input is disabled at init

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def forward(self, input):
        B, T =  input.size()
        pos = torch.arange(0, T, dtype=torch.long, device=input.device)
        pos_emb = self.neurontransformer.wpe(pos)
        tok_emb = self.neurontransformer.wte(input)
        x = pos_emb + tok_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.neurontransformer.lnf(x)
        y = self.pr(x)
        return y

