from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    p = 3
    while p * p <= n:
        if n % p == 0:
            return False
        p += 2
    return True


def distinct_primes_below(start: int, count: int) -> list[int]:
    """Return deterministic, descending prime table sizes."""
    result: list[int] = []
    candidate = int(start)
    while len(result) < count:
        while candidate >= 2 and not is_prime(candidate):
            candidate -= 1
        if candidate < 2:
            raise ValueError("not enough positive prime capacities")
        result.append(candidate)
        candidate -= 1
    return result


def engram_parameter_plan(*, budget: int, hidden_dim: int = 512,
                          embed_per_order: int = 256,
                          heads_per_order: int = 8,
                          insertion_layers: Sequence[int] = (1, 4),
                          kernel: int = 4) -> dict:
    """Allocate distinct per-head prime tables without exceeding ``budget``.

    The source configuration has two orders and eight heads per order.  The
    tables at every insertion site use the same capacities; controls are
    counted exactly and the returned ``actual_parameters`` is a hard upper
    bound for the trainable Engram branch.
    """
    if embed_per_order % heads_per_order:
        raise ValueError("embed_per_order must divide heads_per_order")
    if not insertion_layers or len(set(insertion_layers)) != len(insertion_layers):
        raise ValueError("insertion layers must be distinct")
    per_head = embed_per_order // heads_per_order
    memory_width = 2 * embed_per_order
    # W_K, W_V; query/key/memory RMSNorm scales; depthwise convolution.
    controls_per_layer = (2 * hidden_dim * memory_width + 3 * hidden_dim
                          + hidden_dim * kernel)
    controls = len(insertion_layers) * controls_per_layer
    remaining = int(budget) - controls
    if remaining <= 0:
        raise ValueError("budget cannot fit Engram controls")
    max_sum = remaining // (len(insertion_layers) * per_head)
    average = max_sum // (2 * heads_per_order)
    # Use a short consecutive prime window centered around the budget.  This
    # keeps all heads at essentially the same capacity while recovering the
    # unused rows left by simply taking the primes below ``average``.
    n_heads = 2 * heads_per_order
    best: tuple[int, list[int]] | None = None
    for start in range(max(2, average - 2048), average + 2049):
        try:
            candidate = distinct_primes_below(start + n_heads - 1, n_heads)
        except ValueError:
            # Very small CPU smoke budgets may not have sixteen positive
            # primes in the first candidate window. Continue at larger starts.
            continue
        total = sum(candidate)
        if total <= max_sum and (best is None or total > best[0]):
            best = (total, candidate)
    if best is None:
        raise ValueError("could not allocate prime capacities below budget")
    capacities_one_site = best[1]
    table_parameters = (len(insertion_layers) * per_head
                        * sum(capacities_one_site))
    actual = controls + table_parameters
    while actual > budget:
        # This is normally unnecessary because the floor calculation leaves a
        # margin, but keeps the planner safe if dimensions are changed.
        capacities_one_site[-1] = distinct_primes_below(
            capacities_one_site[-1] - 1, 1)[0]
        capacities_one_site = sorted(set(capacities_one_site), reverse=True)
        if len(capacities_one_site) != 2 * heads_per_order:
            raise ValueError("prime capacity collision while shrinking budget")
        table_parameters = len(insertion_layers) * per_head * sum(capacities_one_site)
        actual = controls + table_parameters
    return {
        "budget": int(budget),
        "hidden_dim": int(hidden_dim),
        "embed_per_order": int(embed_per_order),
        "heads_per_order": int(heads_per_order),
        "head_dim": int(per_head),
        "max_ngram": 3,
        "insertion_layers": [int(x) for x in insertion_layers],
        "kernel": int(kernel),
        "dilation": 3,
        "controls_per_layer": int(controls_per_layer),
        "control_parameters": int(controls),
        "capacities_one_site": [int(x) for x in capacities_one_site],
        "table_parameters": int(table_parameters),
        "actual_parameters": int(actual),
        "unused_budget": int(budget - actual),
    }


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight.to(x.dtype)


@dataclass(frozen=True)
class EngramConfig:
    vocab_size: int = 8192
    hidden_dim: int = 512
    embed_per_order: int = 256
    heads_per_order: int = 8
    capacities_one_site: tuple[int, ...] = ()
    kernel: int = 4
    dilation: int = 3
    pad_id: int = 0
    layer_id: int = 1
    hash_seed: int = 0

    @property
    def head_dim(self) -> int:
        return self.embed_per_order // self.heads_per_order

    @property
    def memory_width(self) -> int:
        return 2 * self.embed_per_order


class EngramHash:
    """Deterministic multiplicative-XOR 2/3-gram address generator."""

    def __init__(self, config: EngramConfig) -> None:
        self.config = config
        if len(config.capacities_one_site) != 2 * config.heads_per_order:
            raise ValueError("need one capacity for every order/head table")
        max_long = np.iinfo(np.int64).max
        half_bound = max(1, int(max_long // 100000) // 2)
        rng = np.random.default_rng(config.hash_seed + 10007 * config.layer_id)
        self.multipliers = [int(x) for x in (rng.integers(
            0, half_bound, size=3, dtype=np.int64) * 2 + 1).tolist()]

    def __call__(self, ids: Tensor) -> list[Tensor]:
        if ids.ndim != 2:
            raise ValueError("ids must be [batch, time]")
        # CPU int64 arithmetic is deterministic for the D9 token range.  The
        # multipliers are bounded so token*multiplier remains below 2**63.
        x = ids.to(torch.int64)
        shifts = [x]
        for lag in (1, 2):
            shifts.append(F.pad(x, (lag, 0), value=self.config.pad_id)[:, :-lag])
        out: list[Tensor] = []
        for order in (2, 3):
            mix = shifts[0] * self.multipliers[0]
            for lag in range(1, order):
                mix = torch.bitwise_xor(mix, shifts[lag] * self.multipliers[lag])
            for head, capacity in enumerate(self.config.capacities_one_site[(order - 2) * self.config.heads_per_order:
                                                                              (order - 1) * self.config.heads_per_order]):
                out.append(torch.remainder(mix, int(capacity)))
        return out


class EngramReference(nn.Module):
    """Public-equation Engram branch, used for CPU parity and smoke tests."""

    def __init__(self, config: EngramConfig) -> None:
        super().__init__()
        if config.embed_per_order % config.heads_per_order:
            raise ValueError("embed_per_order must divide heads_per_order")
        self.config = config
        self.hash = EngramHash(config)
        self.tables = nn.ModuleList([
            nn.Embedding(int(capacity), config.head_dim)
            for capacity in config.capacities_one_site
        ])
        nn.init.normal_(self.tables[0].weight, std=0.02)
        for table in self.tables[1:]:
            nn.init.normal_(table.weight, std=0.02)
        self.key = nn.Linear(config.memory_width, config.hidden_dim, bias=False)
        self.value = nn.Linear(config.memory_width, config.hidden_dim, bias=False)
        self.query_norm = RMSNorm(config.hidden_dim)
        self.key_norm = RMSNorm(config.hidden_dim)
        self.memory_norm = RMSNorm(config.hidden_dim)
        self.conv = nn.Conv1d(config.hidden_dim, config.hidden_dim, config.kernel,
                              dilation=config.dilation, groups=config.hidden_dim,
                              bias=False)
        # Eq. (5) starts as the gated value residual; this is also the source
        # paper's explicit zero-convolution initialization.
        nn.init.zeros_(self.conv.weight)

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize the branch with the public Engram recipe."""
        for table in self.tables:
            nn.init.normal_(table.weight, std=0.02)
            table.to(dtype=torch.bfloat16)
        scale = (3.0 / self.config.hidden_dim) ** 0.5
        nn.init.uniform_(self.key.weight, -scale, scale)
        nn.init.uniform_(self.value.weight, -scale, scale)
        nn.init.ones_(self.query_norm.weight)
        nn.init.ones_(self.key_norm.weight)
        nn.init.ones_(self.memory_norm.weight)
        nn.init.zeros_(self.conv.weight)

    def retrieve(self, ids: Tensor) -> Tensor:
        addresses = self.hash(ids)
        return torch.cat([table(address) for table, address in zip(self.tables, addresses)], dim=-1)

    def forward(self, hidden: Tensor, ids: Tensor) -> Tensor:
        if hidden.ndim != 3 or ids.ndim != 2 or hidden.shape[:2] != ids.shape:
            raise ValueError("hidden must be [B,T,D] and ids [B,T]")
        memory = self.retrieve(ids)
        key = self.key(memory)
        value = self.value(memory)
        q = self.query_norm(hidden)
        k = self.key_norm(key)
        gate = torch.sigmoid((q * k).sum(-1, keepdim=True) / math.sqrt(self.config.hidden_dim))
        gated = gate * value
        normalized = self.memory_norm(gated).transpose(1, 2)
        pad = (self.config.kernel - 1) * self.config.dilation
        conv = self.conv(F.pad(normalized, (pad, 0)))[..., :hidden.shape[1]]
        return gated + F.silu(conv).transpose(1, 2)

    def parameter_groups(self) -> list[dict]:
        table_ids = {id(p) for table in self.tables for p in table.parameters()}
        return [
            {"group_name": "engram_lookup", "params": [p for table in self.tables for p in table.parameters()]},
            {"group_name": "engram_control", "params": [p for p in self.parameters() if id(p) not in table_ids]},
        ]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

