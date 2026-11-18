"""Chorus, Engram and FactorEngram language models."""
from . import nanochat, golf


def build_model(config, token_counts=None):
    """Build a model; call initialize_model after moving it to the target device."""
    spec = dict(config)
    method = spec.get('method', spec.get('arm', 'chorus'))
    spec['arm'] = {'chorus': 'ng7', 'codem': 'ng7'}.get(method, method)
    if spec['arm'] not in ('ng7', 'engram', 'factorengram', 'plain'):
        raise ValueError(f'Unknown method: {method}')
    backbone = spec.get('backbone', 'nanochat')
    if backbone == 'nanochat':
        model = nanochat.NanochatGPT(spec)
    elif backbone == 'golf':
        model = golf.build_compact(spec)
    else:
        raise ValueError(f'Unknown backbone: {backbone}')
    return model


def initialize_model(model, seed):
    module = golf if isinstance(model, golf.FineWebGPT) else nanochat
    return module.initialize_compact(model, seed)


def setup_optimizer(model, config):
    module = golf if isinstance(model, golf.FineWebGPT) else nanochat
    return module.setup_compact_optimizer(model, config)
