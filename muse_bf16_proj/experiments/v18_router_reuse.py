# okc: problem=muse_bf16_proj
"""Profile-guided projection kernels for H200.

Single-token output/pooler projections use a direct FP32-accumulating GEMV.
Small routers use fused high/low BF16 split-K GEMMs; large routers retain
SGLang's permitted high/low library approach. Weights are prepared once;
every timed invocation computes fresh outputs from its supplied input.
"""

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
    for step in range(K // SPLIT // BK):
        kk = ki + step * BK
        x = tl.load(X + mi[:, None] * K + kk[None, :], mi[:, None] < M, other=0)
        h = tl.load(H + ni[None, :] * K + kk[:, None], ni[None, :] < N, other=0)
        l = tl.load(L + ni[None, :] * K + kk[:, None], ni[None, :] < N, other=0)
        hi = tl.dot(x, h, hi)
        hi = tl.dot(x, l, hi)
    tl.store(P + (part * M + mi[:, None]) * N + ni[None, :], hi,
             (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit
def _reduce(P, O, SIZE: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    sp = tl.arange(0, SPLIT)
    v = tl.load(P + sp[:, None] * SIZE + i[None, :], i[None, :] < SIZE, other=0)
    tl.store(O + i, tl.sum(v, axis=0), i < SIZE)


@triton.jit
def _gemv(X, W, O, N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    kk = tl.arange(0, BLOCK)
    x = tl.load(X + kk, kk < K, other=0).to(tl.float32)
    w = tl.load(W + rows[:, None] * K + kk[None, :],
                (rows[:, None] < N) & (kk[None, :] < K), other=0).to(tl.float32)
    val = tl.sum(w * x[None, :], axis=1)
    tl.store(O + rows, val, rows < N)


def prepare(weights):
    state = dict(weights)
    for name,w in weights.items():
        if name != 'router':
            state[name+'_t'] = w.T.contiguous()
    hi = weights['router'].to(torch.bfloat16)
    lo = (weights['router'] - hi.float()).to(torch.bfloat16)
    state['hi'] = hi.contiguous()
    state['lo'] = lo.contiguous()
    state['hi_t'] = hi.T.contiguous()
    state['lo_t'] = lo.T.contiguous()
    return state


def kernel_fn(data, state):
    x = data['x']
    c = data['config']
    if x.shape[0] == 1:
        n, k = c['n'], x.shape[1]
        rows = 2 if c['layer'] == 'o' else 1
        out = torch.empty((1,n), device=x.device, dtype=state[c['layer']].dtype)
        _gemv[(triton.cdiv(n,rows),)](x, state[c['layer']], out, n,k,rows,triton.next_power_of_2(k),num_warps=(8 if k>=4096 else 4))
        return out
    if c['layer'] != 'router':
        if c['layer'] == 'pooler' or (c['layer'] == 'unpooler' and x.shape[0]>8):
            return x @ state[c['layer']+'_t'][:,:c['n']]
        return x @ state[c['layer']][:c['n']].T
    m, k = x.shape
    n = c['n']
    split = 32 if m <= 32 else 16
    bm = 16 if m <= 16 else (32 if m <= 32 else 128)
    bn = 32
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    _router[(triton.cdiv(m,bm), triton.cdiv(n,bn), split)](
        x, state['hi'], state['lo'], partial, m,n,k,bm,bn,64,split)
    _reduce[(triton.cdiv(m*n,256),)](partial,out,m*n,split,256)
    return out
