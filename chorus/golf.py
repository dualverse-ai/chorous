from __future__ import annotations
import hashlib
import math
from collections import defaultdict
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from .golf_reference import GPT, Rotary, CastedLinear, Muon, apply_rotary_emb
from .memory_modules import ChorusMemory, OriginalEngram, engram_plan, factorengram_modules
from .memory_optimizer_reference import MuonAdamW


def geometry(config):
    def field(a,b,default): return int(config.get(a,config.get(b,default)))
    return dict(vocab_size=int(config.get('vocab_size',1024)),num_layers=field('n_layer','depth',9),
        model_dim=field('n_embd','width',512),num_heads=field('n_head','query_heads',8),
        num_kv_heads=field('n_kv_head','kv_heads',4),mlp_mult=int(config.get('mlp_expansion',2)),
        tie_embeddings=True,tied_embed_init_std=.005,logit_softcap=30.,rope_base=10000.,qk_gain_init=1.5)


class FineWebGPT(GPT):
    def __init__(self,spec):
        self.compact_config=dict(spec)
        g=geometry(spec)
        super().__init__(**g)
        self.arm={'codem':'ng7','engram_reference':'engram'}.get(spec.get('arm','plain'),spec.get('arm','plain'))
        if self.arm not in ('plain','ng7','engram','factorengram'):
            raise ValueError(f'Unknown arm {self.arm}')
        d,v,depth=g['model_dim'],g['vocab_size'],g['num_layers']
        if d<32 or d%4: raise ValueError('Width must be at least32 and divisible by4')
        budget=int(spec.get('memory_budget',26_000_000))
        self.config=SimpleNamespace(n_embd=d,n_layer=depth,n_head=g['num_heads'],n_kv_head=g['num_kv_heads'],vocab_size=v,
            sequence_len=int(spec.get('sequence_len',spec.get('context',1024))),mode='oe_ember' if self.arm=='ng7' else 'plain',oe_lr=float(spec.get('table_lr',.2)))
        self.fe_l1=float(spec.get('fe_l1',.001))
        self.memory=nn.ModuleDict()
        info={}
        if self.arm=='ng7':
            sites=tuple(i for i in range(depth) if i%2==0)
            repair=spec.get('quality_repair',{'exact_token':True,'adaptive_values':True})
            if (not repair.get('exact_token') or not repair.get('adaptive_values') or
                    any(v for k,v in repair.items() if k not in ('exact_token','adaptive_values'))):
                raise ValueError('Chorus uses the token table and adaptive layer reads')
            self.codem=ChorusMemory(v,d,g['num_kv_heads'],g['num_heads'],budget,sites)
            info=dict(value_layers=list(sites),hash_rows=self.codem.rows,exact_token_parameters=v*d,
                shared_phrase=True,adaptive_values=True,local_two_sided_gates=True,
                phrase_conv_kernel=4,attention_value_gates='32 hidden channels to KV-head gates; counted as added controls',
                lookup_optimizer='Ember')
        elif self.arm=='engram':
            sites=tuple(spec.get('engram_layers',(1,7)))
            caps=engram_plan(v,d,budget,sites)
            self.memory=nn.ModuleDict({str(i):OriginalEngram(v,d,caps,i) for i in sites})
            info=dict(insertion_layers=list(sites),max_order=5,heads_per_order=8,embed_per_order=256,
                capacities_one_site=caps,conv_kernel=4,conv_dilation=5,gate='original sigmoid normalized dot',
                token_id_mapping='identity on SentencePiece1024; no additional token-equivalence compression')
        elif self.arm=='factorengram':
            sites=tuple(spec.get('factorengram_layers',(3,4)))
            self.memory=factorengram_modules(v,d,budget,sites)
            info=dict(insertion_layers=list(sites),max_order=3,coefficient_dim=3*d,memory_dim=3*d,fe_l1=self.fe_l1,
                sparsity_reduction='mean_tokens_mean_insertion_layers_sum_channels')
        if self.memory and (len(self.memory)!=len(sites) or min(sites)<0 or max(sites)>=depth):
            raise ValueError('Memory insertion sites must be distinct valid block indices')
        common=sum(p.numel() for name,p in self.named_parameters() if is_common(name))
        added=sum(p.numel() for name,p in self.named_parameters() if not is_common(name))
        lookup=sum(p.numel() for name,p in self.named_parameters() if not is_common(name) and any(t in name for t in ('.tables.','.oe_tables.','.A.')))
        if added>budget: raise ValueError(f'Added budget exceeded {added}>{budget}')
        self.memory_spec=dict(arm=self.arm,budget=budget,actual_parameters=added,lookup_parameters=lookup,
            control_parameters=added-lookup,common_parameters=common,unused_budget=budget-added,
            geometry=g,native_value_tables=False,tied_embedding_head=True,
            backbone='Parameter Golf',
            insertion_index_convention='zero-based; Engram after blocks1,7; FactorEngram before blocks3,4',
            **info)

    @property
    def oe_tables(self):
        return self.codem.oe_tables if self.arm=='ng7' else ()

    def oe_count(self,idx,counts,cnt_comp=None):
        if self.arm!='ng7': raise ValueError('Only Chorus uses Ember')
        self.codem.oe_count(idx,counts)

    def oe_mfree_step(self,lr,counts,cnt_comp=None,step=1):
        if self.arm!='ng7': raise ValueError('Only Chorus uses Ember')
        self.codem.oe_mfree_step(lr,counts,step)

    def _value_attention(self,attn,normalized,hidden,phrase,site):
        b,t,d=normalized.shape
        q=attn.c_q(normalized).reshape(b,t,attn.num_heads,attn.head_dim).transpose(1,2)
        k=attn.c_k(normalized).reshape(b,t,attn.num_kv_heads,attn.head_dim).transpose(1,2)
        v=attn.c_v(normalized).reshape(b,t,attn.num_kv_heads,attn.head_dim)
        readout,gate=self.codem.value(site,hidden,normalized,phrase)
        v=v+(gate[...,None]*readout.reshape(b,t,attn.num_kv_heads,attn.head_dim)).to(v.dtype)
        v=v.transpose(1,2)
        q=F.rms_norm(q,(q.size(-1),)); k=F.rms_norm(k,(k.size(-1),))
        cos,sin=attn.rotary(t,normalized.device,q.dtype)
        q=apply_rotary_emb(q,cos,sin); k=apply_rotary_emb(k,cos,sin)
        q=q*attn.q_gain.to(q.dtype)[None,:,None,None]
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=None,is_causal=True,enable_gqa=attn.num_kv_heads!=attn.num_heads)
        return attn.proj(y.transpose(1,2).contiguous().reshape(b,t,d))

    def _block(self,i,x,x0,phrase):
        block=self.blocks[i]
        if self.arm!='ng7' or i not in self.codem.sites:
            return block(x,x0)
        mix=block.resid_mix.to(x.dtype)
        x=mix[0][None,None,:]*x+mix[1][None,None,:]*x0
        attention=self._value_attention(block.attn,block.attn_norm(x),x,phrase,i)
        x=x+block.attn_scale.to(x.dtype)[None,None,:]*attention
        return x+block.mlp_scale.to(x.dtype)[None,None,:]*block.mlp(block.mlp_norm(x))

    def forward(self,input_ids,targets=None,reduction='mean'):
        x=self.tok_emb(input_ids); phrase=None
        if self.arm=='ng7':
            phrase,injection=self.codem(input_ids,x)
            x=x+injection.to(x.dtype)
        x=F.rms_norm(x,(x.size(-1),)); x0=x
        skips=[]; l1=[]
        for i in range(len(self.blocks)):
            if i>=self.num_encoder_layers and skips:
                j=i-self.num_encoder_layers
                x=x+self.skip_weights[j].to(x.dtype)[None,None,:]*skips.pop()
            if self.arm=='factorengram' and str(i) in self.memory:
                delta,sparse=self.memory[str(i)](x,input_ids); x=x+delta; l1.append(sparse)
            x=self._block(i,x,x0,phrase)
            if self.arm in ('engram',) and str(i) in self.memory:
                x=x+self.memory[str(i)](x,input_ids).to(x.dtype)
            if i<self.num_encoder_layers: skips.append(x)
        hidden=self.final_norm(x)
        logits=F.linear(hidden,self.tok_emb.weight)
        logits=(self.logit_softcap*torch.tanh(logits/self.logit_softcap)).float()
        if targets is None: return logits
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),targets.reshape(-1),ignore_index=-1,reduction=reduction)
        if self.training and reduction=='mean' and l1:
            loss=loss+self.fe_l1*torch.stack(l1).mean()
        return loss


def is_common(name):
    return name.startswith(('tok_emb.','blocks.','final_norm.')) or name=='skip_weights'


def common_parameter_names(model):
    return [name for name,_ in model.named_parameters() if is_common(name)]


def build_compact(config,token_counts=None):
    return FineWebGPT(dict(config))


@torch.no_grad()
def initialize_compact(model,seed):
    device=model.tok_emb.weight.device
    if device.type=='meta': raise ValueError('Materialize before initialization')
    # Restore nonpersistent RoPE buffers after to_empty.
    for block in model.blocks:
        rotary=block.attn.rotary
        rotary.inv_freq.copy_(1./(10000.**(torch.arange(0,block.attn.head_dim,2,device=device,dtype=torch.float32)/block.attn.head_dim)))
        rotary._seq_len_cached=0; rotary._cos_cached=None; rotary._sin_cached=None
    # Independent name-based random streams preserve matched common tensors
    # across arms/capacities while using native Golf distributions/constants.
    for name,p in model.named_parameters():
        if not is_common(name): continue
        digest=hashlib.sha256(f'fineweb-golf-common-v1:{seed}:{name}'.encode()).digest()
        gen=torch.Generator(device='cpu').manual_seed(int.from_bytes(digest[:8],'little')%(2**63-1))
        if name=='tok_emb.weight':
            temp=torch.empty(p.shape,device='cpu',dtype=torch.float32)
            nn.init.normal_(temp,std=.005,generator=gen);p.copy_(temp)
        elif name=='skip_weights' or name.endswith(('.attn_scale','.mlp_scale')): p.fill_(1.)
        elif name.endswith('.resid_mix'): p[0].fill_(1.); p[1].zero_()
        elif name.endswith('.q_gain'): p.fill_(1.5)
        elif name.endswith('.proj.weight'): p.zero_()
        else:
            bound=1/math.sqrt(p.shape[-1])
            temp=torch.empty(p.shape,device='cpu',dtype=torch.float32)
            nn.init.uniform_(temp,-bound,bound,generator=gen);p.copy_(temp)
    devices=[device.index or 0] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if model.arm=='ng7': model.codem.reset_parameters()
        else:
            for module in model.memory.values(): module.reset_parameters()
    # Native Golf mixed precision: embeddingsBF16, linear weightsFP32,
    # low-dimensional controlsFP32; memory tables BF16.
    if device.type=='cuda' or model.compact_config.get('cpu_bfloat16',False):
        model.tok_emb.to(dtype=torch.bfloat16)
        if model.arm=='ng7':
            for table in model.codem.oe_tables: table.to(dtype=torch.bfloat16)
        else:
            for module in model.memory.values():
                for child in module.modules():
                    if isinstance(child,nn.Embedding): child.to(dtype=torch.bfloat16)
    model.memory_spec['initialization_seed']=int(seed)
    model.memory_spec['common_initialization']='CPU float32 per-name RNG, copied to device; same native distributions, exact shared tensors across arms/devices'
    return model


class OptimizerBundle:
    """Backbone optimizers and method-specific memory optimizers.

    Ember counts/step are called by the trainer after this bundle's step.
    Parameter groups share references with their underlying optimizers, so
    the existing schedule mutates the actual learning rates/momenta.
    """
    def __init__(self,optimizers,extra_groups=()):
        self.optimizers=optimizers
        self.param_groups=[g for opt in optimizers for g in opt.param_groups]+list(extra_groups)
    @property
    def state(self):
        return {p:state for opt in self.optimizers for p,state in opt.state.items()}
    def step(self):
        for optimizer in self.optimizers: optimizer.step()
    def zero_grad(self,set_to_none=True):
        for group in self.param_groups:
            for p in group['params']:
                if set_to_none:p.grad=None
                elif p.grad is not None:p.grad.zero_()
    def state_dict(self): return {'optimizers':[opt.state_dict() for opt in self.optimizers]}
    def load_state_dict(self,state):
        if len(state['optimizers'])!=len(self.optimizers): raise ValueError('Optimizer bundle mismatch')
        for opt,s in zip(self.optimizers,state['optimizers']):opt.load_state_dict(s)
        self.param_groups=[g for opt in self.optimizers for g in opt.param_groups]+[g for g in self.param_groups if g['kind']=='oe_mfree']


def setup_compact_optimizer(model,config):
    groups=[]; matrices=[]; scalars=[]
    for name,p in model.blocks.named_parameters():
        if p.ndim==2 and 'resid_mix' not in name:matrices.append(p)
        else:scalars.append(p)
    scalars.append(model.skip_weights)
    fused=model.tok_emb.weight.device.type=='cuda'
    def adam(name,params,lr):
        return torch.optim.Adam([dict(group_name=name,params=list(params),kind='adam',lr=lr,initial_lr=lr,base_lr=lr)],betas=(.9,.95),eps=1e-8,fused=fused)
    optimizers=[adam('native_tied_embedding',[model.tok_emb.weight],.05),adam('native_scalars',scalars,.04)]
    common_muon=Muon(matrices,lr=.04,momentum=.85,backend_steps=5)
    for g in common_muon.param_groups:g.update(kind='muon',group_name='native_golf_muon',optimizer_family='native_golf_muon',initial_lr=.04,base_lr=.04,momentum_warmup_steps=500,momentum_start=.85,momentum_end=.95)
    optimizers.append(common_muon)
    added_groups=[]; memory_matrices=defaultdict(list);tn_native_matrices=[]
    def add(name,params,lr=.04,kind='adamw',**options):
        params=list(params)
        if not params:return
        defaults=dict(betas=(.9,.95),eps=1e-8,weight_decay=0.) if kind=='adamw' else dict(momentum=.85,ns_steps=5,beta2=.95,weight_decay=float(config.get('weight_decay',.2)),momentum_start=.85,momentum_end=.95,momentum_warmup_steps=500)
        added_groups.append(dict(group_name=name,params=params,kind=kind,lr=lr,initial_lr=lr,base_lr=lr,**defaults,**options))
    extra=[]
    if model.arm=='ng7':
        for module in (model.codem.oe_proj,model.codem.bcd_ctx_proj,*model.codem.pve_projs.values(),*model.codem.value_conditioners.values(),*model.codem.ve_gates.values()):
            memory_matrices[tuple(module.weight.shape)].append(module.weight)
        add('codem_convolution',model.codem.phrase_dw_conv.parameters())
        lr=float(config.get('table_lr',.2))*(model.config.n_embd/768)**-.5
        extra=[dict(group_name='ember_lookup',params=list(model.codem.oe_tables.parameters()),kind='oe_mfree',lr=lr,initial_lr=lr,base_lr=lr)]
    elif model.arm=='engram':
        for i,module in model.memory.items():
            # Frozen Engram port: table AdamW.05; every other added
            # parameter (including projections) in AdamW.04.
            add('engram_lookup_'+i,module.tables.parameters(),float(config.get('engram_embed_lr',.05)))
            table_ids={id(p) for p in module.tables.parameters()}
            add('engram_controls_'+i,[p for p in module.parameters() if id(p) not in table_ids],float(config.get('engram_ctrl_lr',.04)))
    elif model.arm=='factorengram':
        for i,module in model.memory.items():
            for g in module.parameter_groups():
                lr=float(config.get('table_lr',.001)) if g['group_name'].endswith('lookup') else float(config.get('control_lr',.01))
                add(g['group_name']+'_'+i,g['params'],lr)
    for shape,params in sorted(memory_matrices.items()):add('memory_muon_'+'x'.join(map(str,shape)),params,kind='muon')
    if added_groups:optimizers.append(MuonAdamW(added_groups))
    bundle=OptimizerBundle(optimizers,extra)
    covered=[id(p) for g in bundle.param_groups for p in g['params']]
    expected={id(p) for p in model.parameters() if p.requires_grad}
    if len(covered)!=len(set(covered)) or set(covered)!=expected:raise AssertionError('Optimizer coverage not unique/exhaustive')
    model.memory_spec['optimizer_recipe']=dict(common='exact native Golf Adam+NewtonSchulz Muon; no weight decay',
        memory='Chorus Muon/Ember; baseline AdamW',
        native_scalar_adam_lr=.04,native_tied_adam_lr=.05,native_muon_lr=.04,
        all_parameters_exactly_once=True,groups=[{k:v for k,v in g.items() if k!='params'} for g in bundle.param_groups])
    return bundle
