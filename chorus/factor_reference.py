from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class _AffineRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)

@dataclass(frozen=True)
class FactorEngramConfig:
    vocab_size: int
    hidden_dim: int
    coefficient_dim: int
    memory_dim: int
    bigram_capacity: int
    trigram_capacity: int
    heads_per_order: tuple[int, int, int] = (4, 4, 4)
    conv_kernel: int = 4
    conv_dilation: int = 1
    rms_eps: float = 1e-6
    coefficient_init_std: float = 0.02

class FactorEngramLayer(nn.Module):
    """Independent pre-attention FactorEngram branch; returns (delta, raw L1).

    L1 is sum over coefficient channels, mean over token/batch dimensions.
    Average it over insertion layers and multiply by lambda outside the model.
    """

    def __init__(self, config: FactorEngramConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        if len(c.heads_per_order) != 3 or any(h <= 0 for h in c.heads_per_order):
            raise ValueError("FactorEngram must retrieve all orders 1, 2, 3")
        if any(v <= 0 for v in (c.vocab_size, c.hidden_dim, c.memory_dim,
                                c.bigram_capacity, c.trigram_capacity,
                                c.coefficient_dim, c.conv_kernel, c.conv_dilation)):
            raise ValueError("Dimensions, capacities, kernel and dilation must be positive")
        heads = sum(c.heads_per_order)
        if c.coefficient_dim % heads:
            raise ValueError("coefficient_dim must be divisible by total branch count")
        width = c.coefficient_dim // heads
        self.orders: list[int] = []
        self.tables = nn.ModuleList()
        for order, count in enumerate(c.heads_per_order, start=1):
            for _ in range(count):
                branch = len(self.orders)
                capacity = (c.vocab_size if order == 1 else
                            (c.bigram_capacity if order == 2 else c.trigram_capacity) + 2 * branch)
                self.orders.append(order)
                self.tables.append(nn.Embedding(capacity, width))
        self.dictionary = nn.Parameter(torch.empty(c.coefficient_dim, c.memory_dim))
        self.query = nn.Linear(c.hidden_dim, c.memory_dim, bias=False)
        self.query_norm = _AffineRMSNorm(c.memory_dim, c.rms_eps)
        self.value = nn.Linear(c.memory_dim, c.hidden_dim, bias=False)
        self.output_norm = _AffineRMSNorm(c.hidden_dim, c.rms_eps)
        self.conv = nn.Conv1d(c.hidden_dim, c.hidden_dim, c.conv_kernel,
                              dilation=c.conv_dilation, groups=c.hidden_dim, bias=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for table in self.tables:
            nn.init.normal_(table.weight, std=self.config.coefficient_init_std)
        nn.init.normal_(self.dictionary, std=self.config.memory_dim ** -0.5)
        self.query.reset_parameters()
        self.value.reset_parameters()
        nn.init.ones_(self.query_norm.weight)
        nn.init.ones_(self.output_norm.weight)
        nn.init.zeros_(self.conv.weight)

    def retrieve_coefficients(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [B,T]")
        ids = token_ids.long()
        length = ids.shape[1]
        suffixes = {1: ids}
        for order in (2, 3):
            lagged = F.pad(ids, (order - 1, 0), value=self.config.vocab_size)[:, :length]
            suffixes[order] = (suffixes[order - 1] + lagged *
                               ((self.config.vocab_size + 1) ** (order - 1)))
        pieces = []
        for order, table in zip(self.orders, self.tables):
            address = ids if order == 1 else suffixes[order].remainder(table.num_embeddings)
            pieces.append(table(address))
        return torch.cat(pieces, dim=-1)

    def forward(self, hidden: Tensor, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        z = self.retrieve_coefficients(token_ids)
        q = self.query_norm(self.query(hidden.to(self.query.weight.dtype)))
        # Dictionary tied between contextual scores and reconstruction, eqs. 7/9.
        gates = torch.sigmoid(F.linear(q, self.dictionary) / math.sqrt(self.config.hidden_dim))
        reconstructed = (z * gates.to(z.dtype)) @ self.dictionary
        v = self.value(reconstructed)
        normalized = self.output_norm(v).transpose(1, 2).to(self.conv.weight.dtype)
        pad = (self.config.conv_kernel - 1) * self.config.conv_dilation
        refinement = F.silu(self.conv(F.pad(normalized, (pad, 0)))).transpose(1, 2)
        delta = v + refinement.to(v.dtype)
        sparsity = z.float().abs().sum(dim=-1).mean()
        return delta.to(hidden.dtype), sparsity

    def parameter_groups(self) -> list[dict]:
        lookup = [table.weight for table in self.tables]
        lookup_ids = {id(p) for p in lookup}
        return [dict(group_name="factorengram_lookup", params=lookup),
                dict(group_name="factorengram_control",
                     params=[p for p in self.parameters() if id(p) not in lookup_ids])]
