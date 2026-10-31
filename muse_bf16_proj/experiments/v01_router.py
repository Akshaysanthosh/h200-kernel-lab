# okc: problem=muse_bf16_proj
"""Starter: the projections in plain PyTorch, correct but slow.

Implement kernel_fn, and prepare to repack the weights (called once, untimed)."""

from typing import TypedDict

import torch
import triton
import triton.language as tl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


@triton.jit
def _router(X, H, L, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    ki = part * (K // SPLIT) + tl.arange(0, BK)
    hi = tl.full((BM, BN), 0, tl.float32)
    lo = tl.full((BM, BN), 0, tl.float32)
    for step in range(K // SPLIT // BK):
        kk = ki + step * BK
        x = tl.load(X + mi[:, None] * K + kk[None, :], mi[:, None] < M, other=0)
        h = tl.load(H + ni[None, :] * K + kk[:, None], ni[None, :] < N, other=0)
        l = tl.load(L + ni[None, :] * K + kk[:, None], ni[None, :] < N, other=0)
        hi = tl.dot(x, h, hi)
        lo = tl.dot(x, l, lo)
    tl.store(P + (part * M + mi[:, None]) * N + ni[None, :], hi + lo,
             (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit
def _reduce(P, O, SIZE: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    sp = tl.arange(0, SPLIT)
    v = tl.load(P + sp[:, None] * SIZE + i[None, :], i[None, :] < SIZE, other=0)
    tl.store(O + i, tl.sum(v, axis=0), i < SIZE)


def prepare(weights):
    state = dict(weights)
    hi = weights['router'].to(torch.bfloat16)
    lo = (weights['router'] - hi.float()).to(torch.bfloat16)
    state['hi'] = hi.contiguous()
    state['lo'] = lo.contiguous()
    return state


def kernel_fn(data, state):
    x = data['x']
    c = data['config']
    if c['layer'] != 'router':
        return x @ state[c['layer']][:c['n']].T
    m, k = x.shape
    n = c['n']
    split = 8
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    _router[(triton.cdiv(m,16), triton.cdiv(n,32), split)](
        x, state['hi'], state['lo'], partial, m,n,k,16,32,64,split)
    _reduce[(triton.cdiv(m*n,256),)](partial,out,m*n,split,256)
    return out
