from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .golf_reference import CastedLinear
from .factor_reference import FactorEngramConfig, FactorEngramLayer, _AffineRMSNorm

def norm(x):
    with torch.autocast(device_type=x.device.type, enabled=False):
        return F.rms_norm(x, (x.size(-1),))

def previous(x, lag, pad=0):
    return F.pad(x, (lag, 0), value=pad)[:, :x.shape[1]]

class ChorusMemory(nn.Module):

    def __init__(self, vocab, width, kv_heads, heads, budget, sites):
        super().__init__()
        self.vocab, self.width, self.sites = (vocab, width, tuple(sites))
        kv = kv_heads * (width // heads)
        controls = 3 * width * width + len(sites) * (width * kv + width * width + 32 * kv_heads) + 4 * width
        self.rows = (budget - controls - vocab * width) // width - 3
        if self.rows < 1:
            raise ValueError('Chorus table budget too small')
        self.oe_tables = nn.ModuleList([nn.Embedding(self.rows + 2 * j, width // 4) for j in range(4)] + [nn.Embedding(vocab, width)])
        self.oe_proj = CastedLinear(width, width, bias=False)
        self.bcd_ctx_proj = CastedLinear(width, 2 * width, bias=False)
        self.phrase_dw_conv = nn.Conv1d(width, width, 4, groups=width, bias=False)
        self.pve_projs = nn.ModuleDict({str(i): CastedLinear(width, kv, bias=False) for i in sites})
        self.value_conditioners = nn.ModuleDict({str(i): CastedLinear(width, width, bias=False) for i in sites})
        self.ve_gates = nn.ModuleDict({str(i): CastedLinear(32, kv_heads, bias=False) for i in sites})
        for j, t in enumerate(self.oe_tables):
            self.register_buffer(f'ember_r_{j}', torch.zeros(t.num_embeddings), persistent=False)
            self.register_buffer(f'ember_c_{j}', torch.zeros(t.embedding_dim), persistent=False)

    @torch.no_grad()
    def reset_parameters(self):
        for table in self.oe_tables:
            nn.init.zeros_(table.weight)
        nn.init.eye_(self.oe_proj.weight)
        s = 0.25 * math.sqrt(3 / self.width)
        nn.init.uniform_(self.bcd_ctx_proj.weight, -s, s)
        nn.init.zeros_(self.phrase_dw_conv.weight)
        for projection in self.pve_projs.values():
            nn.init.eye_(projection.weight)
            projection.weight.mul_(0.25)
        for gate in [*self.value_conditioners.values(), *self.ve_gates.values()]:
            nn.init.zeros_(gate.weight)
        for name, buf in self.named_buffers():
            if name.startswith('ember_'):
                buf.zero_()

    def oe_ids(self, idx):
        ids = idx.long()
        out = []
        for order in (2, 3):
            ids = ids + previous(idx, order - 1, self.vocab).long() * (self.vocab + 1) ** (order - 1)
            for head in range(2):
                out.append(ids.remainder(self.rows + 2 * ((order - 2) * 2 + head)))
        return out + [idx.long()]

    def forward(self, idx, word):
        ids = self.oe_ids(idx)
        raw = torch.cat([self.oe_tables[j](ids[j]) for j in range(4)], -1) + self.oe_tables[4](idx)
        w0 = norm(word.detach())
        w1 = F.pad(torch.roll(w0, 1, -1), (0, 0, 1, 0))[:, :word.shape[1]]
        w2 = F.pad(torch.roll(w0, 2, -1), (0, 0, 2, 0))[:, :word.shape[1]]
        context = (w0 + 0.70710678 * w1 + 0.5 * w2) * 0.57735027
        pre, post = (2 * torch.sigmoid(self.bcd_ctx_proj(context))).chunk(2, -1)
        phrase = raw * pre.to(raw.dtype)
        conv = self.phrase_dw_conv(F.pad(norm(phrase).transpose(1, 2).to(self.phrase_dw_conv.weight.dtype), (3, 0))).transpose(1, 2)
        phrase = phrase + F.silu(conv).to(phrase.dtype)
        injection = self.oe_proj(phrase) * post
        return (phrase, injection)

    def value(self, site, hidden, normalized_hidden, phrase):
        key = str(site)
        h = F.rms_norm(hidden.float(), (self.width,), eps=1e-05)
        adaptive = 2 * torch.sigmoid(self.value_conditioners[key](h))
        readout = self.pve_projs[key](phrase * adaptive.to(phrase.dtype))
        gate = 2 * torch.sigmoid(self.ve_gates[key](normalized_hidden[..., :32]))
        return (readout, gate)

    @torch.no_grad()
    def oe_count(self, idx, counts):
        for count, address in zip(counts, self.oe_ids(idx)):
            count.index_add_(0, address.reshape(-1), torch.ones(address.numel(), device=idx.device, dtype=torch.float32))

    @torch.no_grad()
    def oe_mfree_step(self, lr, counts, step=1):
        beta2 = 0.95
        for j, table in enumerate(self.oe_tables):
            if table.weight.grad is None:
                continue
            count = counts[j]
            hit = count > 0
            g = table.weight.grad.float()
            r = getattr(self, f'ember_r_{j}')
            c = getattr(self, f'ember_c_{j}')
            r.lerp_(g.square().mean(-1), 1 - beta2)
            c.lerp_(g.square().mean(0), 1 - beta2)
            if hit.any():
                bias = 1 - beta2 ** step
                rh = (r[hit] / bias).clamp_min(1e-30)
                ch = (c / bias).clamp_min(1e-30)
                cn = ch / ch.mean().clamp_min(1e-30)
                denom = (rh[:, None] * cn[None, :]).sqrt() + 1e-10
                u = g[hit] / denom
                table.weight[hit] = (table.weight[hit].float() - lr * u).to(table.weight.dtype)
            count.zero_()

def prime_below(n):
    while n >= 2:
        if n == 2 or (n % 2 and all((n % d for d in range(3, math.isqrt(n) + 1, 2)))):
            return n
        n -= 1
    raise ValueError('Cannot allocate requested prime tables')

def engram_plan(vocab, width, budget, sites, heads_per_order=8, embed_per_order=256):
    memory_width = 4 * embed_per_order
    head_dim = embed_per_order // heads_per_order
    controls = len(sites) * (2 * width * memory_width + 3 * width + 4 * width)
    avg = (budget - controls) // (len(sites) * head_dim * 4 * heads_per_order)
    if avg < 140:
        raise ValueError('Engram budget too small for32 distinct prime heads')
    count = 4 * heads_per_order
    max_sum = (budget - controls) // (len(sites) * head_dim)
    primes = []
    candidate = avg + 2048
    while candidate >= max(2, avg - 2048):
        prime = prime_below(candidate)
        primes.append(prime)
        candidate = prime - 1
    windows = [primes[i:i + count] for i in range(len(primes) - count + 1)]
    feasible = [w for w in windows if sum(w) <= max_sum]
    if not feasible:
        raise ValueError('No prime window below budget')
    return max(feasible, key=sum)

class OriginalEngram(nn.Module):

    def __init__(self, vocab, width, capacities, site, seed=0):
        super().__init__()
        self.vocab, self.width = (vocab, width)
        self.capacities = list(capacities)
        self.tables = nn.ModuleList([nn.Embedding(c, 32) for c in capacities])
        memory_width = 32 * len(capacities)
        self.key = CastedLinear(memory_width, width, bias=False)
        self.value = CastedLinear(memory_width, width, bias=False)
        self.query_norm = _AffineRMSNorm(width, 1e-06)
        self.key_norm = _AffineRMSNorm(width, 1e-06)
        self.memory_norm = _AffineRMSNorm(width, 1e-06)
        self.conv = nn.Conv1d(width, width, 4, dilation=5, groups=width, bias=False)
        rng = np.random.default_rng(seed + 10007 * site)
        bound = int(np.iinfo(np.int64).max // 100000) // 2
        self.multipliers = [int(x) for x in rng.integers(0, bound, size=5, dtype=np.int64) * 2 + 1]

    @torch.no_grad()
    def reset_parameters(self):
        for table in self.tables:
            nn.init.normal_(table.weight, std=0.02)
        scale = math.sqrt(3 / self.width)
        nn.init.uniform_(self.key.weight, -scale, scale)
        nn.init.uniform_(self.value.weight, -scale, scale)
        for module in [self.query_norm, self.key_norm, self.memory_norm]:
            module.weight.fill_(1.0)
        self.conv.weight.zero_()

    def addresses(self, idx):
        shifts = [previous(idx.long(), i, 0) for i in range(5)]
        out = []
        for order in range(2, 6):
            h = shifts[0] * self.multipliers[0]
            for lag in range(1, order):
                h = torch.bitwise_xor(h, shifts[lag] * self.multipliers[lag])
            out.extend((h.remainder(c) for c in self.capacities[(order - 2) * 8:(order - 1) * 8]))
        return out

    def forward(self, hidden, idx):
        memory = torch.cat([t(a) for t, a in zip(self.tables, self.addresses(idx))], -1)
        key, value = (self.key(memory), self.value(memory))
        gate = torch.sigmoid((self.query_norm(hidden) * self.key_norm(key)).sum(-1, keepdim=True) / math.sqrt(self.width))
        u = gate * value
        v = self.memory_norm(u).transpose(1, 2).to(self.conv.weight.dtype)
        return u + F.silu(self.conv(F.pad(v, (15, 0)))).transpose(1, 2).to(u.dtype)

def factorengram_modules(vocab, width, budget, sites):
    coefficient = memory = 3 * width
    if coefficient % 12:
        raise ValueError('FactorEngram width must divide twelve heads')
    branch = coefficient // 12
    fixed = len(sites) * (4 * vocab * branch + 120 * branch + coefficient * memory + 2 * width * memory + memory + width + 4 * width)
    units = (budget - fixed) // (4 * branch)
    base, extra = divmod(units, 2 * len(sites))
    if base < 1:
        raise ValueError('FactorEngram budget too small')
    capacities = [base + (i < extra) for i in range(2 * len(sites))]
    return nn.ModuleDict({str(site): FactorEngramLayer(FactorEngramConfig(vocab, width, coefficient, memory, capacities[2 * j], capacities[2 * j + 1])) for j, site in enumerate(sites)})
