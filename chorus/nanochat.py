"""Nanochat backbone and memory methods used for ClimbMix."""
from __future__ import annotations
import hashlib
import math
from collections import defaultdict
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from .nanochat_blocks import Block, has_ve
from .memory_modules import norm, ChorusMemory, factorengram_modules
from .engram import EngramConfig, EngramReference, engram_parameter_plan
from .memory_optimizer_reference import MuonAdamW


class NanochatGPT(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.compact_config = dict(spec)
        self.arm = spec.get('arm', 'ng7')
        def field(a, b, default): return int(spec.get(a, spec.get(b, default)))
        self.config = c = SimpleNamespace(n_layer=field('n_layer', 'depth', 9),
            n_embd=field('n_embd', 'width', 512), n_head=field('n_head', 'query_heads', 8),
            n_kv_head=field('n_kv_head', 'kv_heads', 4), vocab_size=int(spec.get('vocab_size', 8192)),
            sequence_len=field('sequence_len', 'context', 1024),
            window_pattern=spec.get('window_pattern', 'SSSL'), oe_lr=float(spec.get('table_lr', .2)),
            mode='oe_ember' if self.arm == 'ng7' else 'plain')
        d, depth, vocab = c.n_embd, c.n_layer, c.vocab_size
        if d < 32 or d % 4 or d % c.n_head or c.n_head % c.n_kv_head:
            raise ValueError('Width and attention heads are incompatible')
        self.transformer = nn.ModuleDict({'wte': nn.Embedding(vocab, d),
            'h': nn.ModuleList([Block(c, i) for i in range(depth)])})
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.resid_lambdas = nn.Parameter(torch.ones(depth))
        self.x0_lambdas = nn.Parameter(torch.zeros(depth))
        self.sites = tuple(i for i in range(depth) if has_ve(i, depth))
        kv = c.n_kv_head * (d // c.n_head)
        pattern = c.window_pattern.upper()
        windows = {'S': (c.sequence_len // 2, 0), 'L': (c.sequence_len, 0)}
        self.window_sizes = [windows[pattern[i % len(pattern)]] for i in range(depth)]
        self.window_sizes[-1] = (c.sequence_len, 0)
        self.register_buffer('cos', torch.empty(1, c.sequence_len * 10, 1, d // c.n_head // 2), persistent=False)
        self.register_buffer('sin', torch.empty_like(self.cos), persistent=False)
        budget = int(spec.get('memory_budget', 76_443_648))
        self.oe_tables = nn.ModuleList()
        self.fe_l1 = float(spec.get('fe_l1', .001))
        self.table_lr = float(spec.get('table_lr', .001))
        self.control_lr = float(spec.get('control_lr', .01))
        if self.arm == 'ng7':
            controls = 3*d*d + len(self.sites)*(d*kv+d*d) + 4*d
            lookup = int(spec.get('lookup_parameters', budget-controls))
            self.rows = (lookup-vocab*d)//d-3
            if self.rows < 1: raise ValueError('Table budget is too small')
            c.oe_m = self.rows
            self.vocab, self.width = vocab, d
            self.oe_tables = nn.ModuleList([nn.Embedding(self.rows+2*j, d//4) for j in range(4)] + [nn.Embedding(vocab,d)])
            self.oe_proj = nn.Linear(d,d,bias=False)
            self.bcd_ctx_proj = nn.Linear(d,2*d,bias=False)
            self.phrase_dw_conv = nn.Conv1d(d,d,4,groups=d,bias=False)
            self.pve_projs = nn.ModuleDict({str(i):nn.Linear(d,kv,bias=False) for i in self.sites})
            self.value_conditioners = nn.ModuleDict({str(i):nn.Linear(d,d,bias=False) for i in self.sites})
            for j,t in enumerate(self.oe_tables):
                self.register_buffer(f'ember_r_{j}',torch.zeros(t.num_embeddings),persistent=False)
                self.register_buffer(f'ember_c_{j}',torch.zeros(t.embedding_dim),persistent=False)
        elif self.arm == 'engram':
            sites = tuple(spec.get('engram_layers',(1,4)))
            caps = engram_parameter_plan(budget=budget,hidden_dim=d,insertion_layers=sites)['capacities_one_site']
            self.engram_modules = nn.ModuleDict({str(i):EngramReference(EngramConfig(vocab_size=vocab,
                hidden_dim=d,capacities_one_site=tuple(caps),layer_id=i)) for i in sites})
        elif self.arm == 'factorengram':
            default = ((depth-1)//2-1, (depth-1)//2) if depth > 2 else (0,1)
            self.factorengram = factorengram_modules(vocab,d,budget,tuple(spec.get('factorengram_layers',default)))
        elif self.arm != 'plain': raise ValueError(f'Unknown method: {self.arm}')
        common = set(common_parameter_names(self))
        added = [(n,p) for n,p in self.named_parameters() if n not in common]
        actual = sum(p.numel() for _,p in added)
        if actual > budget: raise ValueError('Added parameters exceed table budget')
        lookup = sum(p.numel() for n,p in added if n.startswith('oe_tables.') or '.tables.' in n)
        self.memory_spec = dict(arm=self.arm,budget=budget,actual_parameters=actual,
            lookup_parameters=lookup,control_parameters=actual-lookup,
            common_parameters=sum(p.numel() for n,p in self.named_parameters() if n in common))

    oe_ids = ChorusMemory.oe_ids
    oe_mfree_step_impl = ChorusMemory.oe_mfree_step

    def oe_count(self, idx, counts, cnt_comp=None):
        return ChorusMemory.oe_count(self, idx, counts)

    def oe_mfree_step(self, lr, counts, cnt_comp=None, step=1):
        return self.oe_mfree_step_impl(lr, counts, step)

    def _phrase(self, idx, word):
        addresses = self.oe_ids(idx)
        raw = torch.cat([self.oe_tables[j](addresses[j]) for j in range(4)], -1) + self.oe_tables[4](idx)
        w0 = norm(word.detach())
        w1 = F.pad(torch.roll(w0,1,-1),(0,0,1,0))[:,:idx.shape[1]]
        w2 = F.pad(torch.roll(w0,2,-1),(0,0,2,0))[:,:idx.shape[1]]
        context = (w0 + .70710678*w1 + .5*w2)*.57735027
        pre, post = (2*torch.sigmoid(self.bcd_ctx_proj(context.to(self.bcd_ctx_proj.weight.dtype)))).chunk(2,-1)
        phrase = raw*pre.to(raw.dtype)
        conv = self.phrase_dw_conv(F.pad(norm(phrase).transpose(1,2).to(self.phrase_dw_conv.weight.dtype),(3,0))).transpose(1,2)
        phrase = phrase + F.silu(conv).to(phrase.dtype)
        injection = self.oe_proj(phrase.to(self.oe_proj.weight.dtype))*post
        return phrase, injection

    def forward(self, idx, targets=None, reduction='mean'):
        word = self.transformer.wte(idx)
        phrase = None
        if self.arm == 'ng7':
            phrase, injection = self._phrase(idx,word)
            word = word + injection.to(word.dtype)
        x = norm(word); x0 = x; terms = []
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i]*x + self.x0_lambdas[i]*x0
            key = str(i)
            if self.arm == 'engram' and key in self.engram_modules:
                x = x + self.engram_modules[key](x,idx).to(x.dtype)
            if self.arm == 'factorengram' and key in self.factorengram:
                delta, sparse = self.factorengram[key](x,idx); x = x+delta; terms.append(sparse)
            ve = None
            if i in self.sites:
                kv = self.config.n_kv_head*(self.config.n_embd//self.config.n_head)
                ve = torch.zeros((*idx.shape,kv),device=idx.device,dtype=torch.bfloat16)
                if phrase is not None:
                    gate = self.value_conditioners[key]
                    h = F.rms_norm(x.float(),(self.config.n_embd,),eps=1e-5)
                    gates = 2*torch.sigmoid(gate(h.to(gate.weight.dtype)))
                    adapted = phrase*gates.to(phrase.dtype)
                    projection = self.pve_projs[key]
                    ve = ve + projection(adapted.to(projection.weight.dtype)).to(ve.dtype)
            x = block(x,ve,(self.cos[:,:idx.shape[1]],self.sin[:,:idx.shape[1]]),self.window_sizes[i])
        logits = 15*torch.tanh(self.lm_head(norm(x)).float()/15)
        if targets is None: return logits
        loss = F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1),ignore_index=-1,reduction=reduction)
        if self.training and reduction == 'mean' and terms:
            loss = loss+self.fe_l1*torch.stack(terms).mean()
        return loss


def common_parameter_names(model):
    return [n for n,_ in model.named_parameters() if n.startswith('transformer.') or n in ('resid_lambdas','x0_lambdas')]


@torch.no_grad()
def initialize_compact(model, seed):
    model.lm_head.weight = model.transformer.wte.weight
    device = model.transformer.wte.weight.device
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices), torch.device(device):
        torch.manual_seed(int(seed))
        # Preserve the training initialization's RNG sequence before common
        # tensors receive their per-name streams below.
        nn.init.normal_(model.transformer.wte.weight, std=1.)
        nn.init.normal_(model.lm_head.weight, std=.001)
        d = model.config.n_embd; s = 3**.5*d**-.5
        for block in model.transformer.h:
            for p in (block.attn.c_q.weight,block.attn.c_k.weight,block.attn.c_v.weight,block.mlp.c_fc.weight):
                nn.init.uniform_(p,-s,s)
            block.attn.c_proj.weight.zero_();block.mlp.c_proj.weight.zero_()
        model.transformer.wte.to(dtype=torch.bfloat16)
        if model.arm == 'ng7':
            for j,t in enumerate(model.oe_tables):
                t.weight.zero_();t.to(dtype=torch.bfloat16)
                getattr(model,f'ember_r_{j}').zero_();getattr(model,f'ember_c_{j}').zero_()
            nn.init.eye_(model.oe_proj.weight)
            nn.init.uniform_(model.bcd_ctx_proj.weight,-.25*s,.25*s)
            model.phrase_dw_conv.weight.zero_()
            for p in model.pve_projs.values():nn.init.eye_(p.weight);p.weight.mul_(.25)
            for p in model.value_conditioners.values():p.weight.zero_()
        elif model.arm == 'engram':
            for module in model.engram_modules.values():module.init_weights()
        elif model.arm == 'factorengram':
            for module in model.factorengram.values():
                module.reset_parameters()
                for t in module.tables:t.to(dtype=torch.bfloat16)
    head_dim = model.config.n_embd//model.config.n_head
    freq = 1/(10000**(torch.arange(0,head_dim,2,dtype=torch.float32,device=device)/head_dim))
    angle = torch.outer(torch.arange(model.config.sequence_len*10,dtype=torch.float32,device=device),freq)
    model.cos = angle.cos().bfloat16()[None,:,None,:]; model.sin = angle.sin().bfloat16()[None,:,None,:]
    common = set(common_parameter_names(model))
    for name,p in model.named_parameters():
        if name not in common:continue
        digest = hashlib.sha256(f'compact-common-v1:{int(seed)}:{name}'.encode()).digest()
        gen = torch.Generator(device=p.device).manual_seed(int.from_bytes(digest[:8],'little')%(2**63-1))
        if name == 'transformer.wte.weight':nn.init.normal_(p,std=.02,generator=gen)
        elif name == 'resid_lambdas':p.fill_(1.)
        elif name == 'x0_lambdas':p.fill_(.1)
        elif name.endswith(('attn.c_proj.weight','mlp.c_proj.weight','ve_gate.weight')):p.zero_()
        else:nn.init.uniform_(p,-math.sqrt(3/d),math.sqrt(3/d),generator=gen)
    model.memory_spec['initialization_seed'] = int(seed)
    return model


def setup_compact_optimizer(model, config_dict):
    """Unique optimizer coverage with a single Adam group for the tied weight.

    The trainer must update each Muon group's momentum with
    .85 + .10*min(completed_steps/500,1), and apply global LR scheduling plus
    any lr_warmup_steps multiplier. Ember table updates remain the source
    oe_count/oe_mfree_step contract, outside optimizer.step().
    """
    config = dict(config_dict)
    groups, assigned = ([], set())

    def add(name, params, kind='adamw', lr=0.04, **options):
        params = list(params)
        if not params:
            return
        if any((id(p) in assigned for p in params)):
            raise AssertionError(f'Duplicate optimizer parameter in {name}')
        assigned.update((id(p) for p in params))
        defaults = dict(momentum=0.85, momentum_start=0.85, momentum_end=0.95, momentum_warmup_steps=500, ns_steps=5, beta2=0.95, weight_decay=float(config.get('weight_decay', 0.2))) if kind == 'muon' else dict(betas=(0.9, 0.95), eps=1e-08, weight_decay=0.0)
        groups.append(dict(group_name=name, params=params, kind=kind, lr=float(lr), initial_lr=float(lr), **defaults, **options))
    add('tied_embedding_head', [model.transformer.wte.weight], lr=0.05)
    add('common_scalars', [model.resid_lambdas, model.x0_lambdas])
    matrices = defaultdict(list)
    for name, p in model.transformer.h.named_parameters():
        matrices[tuple(p.shape)].append(p)
    if model.arm == 'ng7':
        for module in (model.oe_proj, model.bcd_ctx_proj, *model.pve_projs.values()):
            matrices[tuple(module.weight.shape)].append(module.weight)
        if model.arm == 'ng7':
            for conditioner in model.value_conditioners.values():
                matrices[tuple(conditioner.weight.shape)].append(conditioner.weight)
        add('ng7_convolution', model.phrase_dw_conv.parameters())
    if model.arm == 'ng7':
        add('ember_lookup', model.oe_tables.parameters(), kind='oe_mfree', lr=model.config.oe_lr * (model.config.n_embd / 768) ** (-0.5))
    if model.arm == 'factorengram':
        modules = model.factorengram.values()
        for site, module in enumerate(modules):
            for group in module.parameter_groups():
                name, params = (group['group_name'], group['params'])
                lr = model.table_lr if name.endswith('lookup') else model.control_lr
                extra = {}
                add(f'{name}_{site}', params, lr=lr, **extra)
    if model.arm == 'engram':
        tables, controls = [], []
        for module in model.engram_modules.values():
            tables.extend(module.tables.parameters())
            table_ids = {id(p) for p in module.tables.parameters()}
            controls.extend(p for p in module.parameters() if id(p) not in table_ids)
        add('engram_tables', tables, lr=float(config.get('engram_embed_lr', .05)))
        add('engram_controls', controls, lr=float(config.get('engram_ctrl_lr', .04)))
    for shape, params in sorted(matrices.items()):
        add('muon_' + 'x'.join(map(str, shape)), params, kind='muon')
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if assigned != expected:
        missed = [name for name, p in model.named_parameters() if id(p) not in assigned]
        raise AssertionError(f'Optimizer coverage mismatch: {missed}')
    optimizer = MuonAdamW(groups)
    return optimizer

