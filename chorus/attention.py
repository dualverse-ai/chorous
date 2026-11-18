"""Causal sliding-window attention for the Nanochat experiments."""
import torch
import torch.nn.functional as F

_masks = {}
_flex = None


def attention(q, k, v, window_size):
    length, window = q.shape[1], window_size[0]
    if q.device.type == 'cuda':
        global _flex
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        if _flex is None:
            _flex = torch.compile(flex_attention, dynamic=False)
        key = (length, window, str(q.device))
        if key not in _masks:
            def allowed(batch, head, query, key):
                return (query >= key) & ((query - key) <= window)
            _masks[key] = create_block_mask(allowed, B=None, H=None, Q_LEN=length,
                                            KV_LEN=length, device=q.device, _compile=True)
        y = _flex(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                  block_mask=_masks[key], enable_gqa=q.shape[2] != k.shape[2])
    else:
        positions = torch.arange(length, device=q.device)
        distance = positions[:, None] - positions[None, :]
        mask = (distance >= 0) & (distance <= window)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                    v.transpose(1, 2), attn_mask=mask, enable_gqa=q.shape[2] != k.shape[2])
    return y.transpose(1, 2)
