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


def _previous_prepare(weights):
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
    bm = 16 if m <= 32 else 64
    bn = 32 if m <= 32 else 64
    partial = torch.empty((split, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    _router[(triton.cdiv(m,bm), triton.cdiv(n,bn), split)](
        x, state['hi'], state['lo'], partial, m,n,k,bm,bn,64,split)
    _reduce[(triton.cdiv(m*n,256),)](partial,out,m*n,split,256)
    return out

# Lossless block exponent packing: 8-bit sign/mantissa plus 4-bit exponent offset.
# Blocks whose exponents do not fit use the original BF16 weights exactly.
@triton.jit
def _packed_load(S,E,B,W,rows,kk,N:tl.constexpr,K:tl.constexpr):
    mask=(rows<N)&(kk<K)
    idx=rows*K+kk
    base=tl.load(B+rows*(K//256)+kk//256,mask,255).to(tl.uint32)
    sm=tl.load(S+idx,mask,0).to(tl.uint32)
    ep=tl.load(E+idx//2,mask,0).to(tl.uint32)
    delta=(ep>>((kk%2)*4))&15
    bits=(sm&127)|((sm&128)<<8)|((base+delta)<<7)
    decoded=bits.to(tl.uint16).to(tl.bfloat16,bitcast=True)
    raw=tl.load(W+idx,mask&(base==255),0)
    return tl.where(base==255,raw,decoded)

@triton.jit
def _packed_vector(X,W,S,E,B,O,N:tl.constexpr,K:tl.constexpr,R:tl.constexpr):
    rows=tl.program_id(0)*R+tl.arange(0,R)
    token=tl.program_id(1);kk=tl.arange(0,K)
    w=_packed_load(S,E,B,W,rows[:,None],kk[None,:],N,K).to(tl.float32)
    x=tl.load(X+token*K+kk).to(tl.float32)
    val=tl.sum(w*x[None,:],1)
    tl.store(O+token*N+rows,val,rows<N)

@triton.jit
def _packed_dot(X,W,S,E,B,P,O,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SK:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM)
    ni=tl.program_id(1)*BN+tl.arange(0,BN)
    part=tl.program_id(2);kk=part*(K//SK)+tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for step in range(K//SK//BK):
        ki=kk+step*BK
        x=tl.load(X+mi[:,None]*K+ki[None,:],mi[:,None]<M,0)
        w=_packed_load(S,E,B,W,ni[:,None],ki[None,:],N,K)
        acc=tl.dot(x,tl.trans(w),acc)
    offset=mi[:,None]*N+ni[None,:]
    if SK==1:tl.store(O+offset,acc,(mi[:,None]<M)&(ni[None,:]<N))
    else:tl.store(P+part*M*N+offset,acc,(mi[:,None]<M)&(ni[None,:]<N))

def _pack_weights(w):
    n,k=w.shape
    bits=w.view(torch.int16).to(torch.int32)&65535
    ex=(bits>>7)&255; blocks=ex.reshape(n,k//256,256)
    lo=blocks.amin(-1); hi=blocks.amax(-1)
    base=torch.where(((hi-lo)<=15)&(lo<255),lo,255).to(torch.uint8).contiguous()
    sm=((bits&127)|((bits>>8)&128)).to(torch.uint8).contiguous()
    d=(ex-lo.repeat_interleave(256,dim=1)).clamp(0,15)
    ep=(d[:,0::2]|(d[:,1::2]<<4)).to(torch.uint8).contiguous()
    return sm,ep,base

def prepare(weights):
    state=_previous_prepare(weights)
    for name in ('pooler','qkv_gate','unpooler','o'):
        state[name+'_packed']=_pack_weights(weights[name])
    return state

def _packed_run(data,state,config,diagnostic=False):
    x=data['x'];c=data['config'];m,k=x.shape;n=c['n'];w=state[c['layer']]
    sm,ep,base=state[c['layer']+'_packed']
    out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
    if config[0]=='vector':
        _,rows,warps=config
        compiled=_packed_vector[(triton.cdiv(n,rows),m)](x,w,sm,ep,base,out,n,k,rows,num_warps=warps)
    else:
        _,bm,bn,bk,sk,warps,stages=config
        scratch=torch.empty((sk,m,n),device=x.device,dtype=torch.float32) if sk>1 else out
        compiled=_packed_dot[(triton.cdiv(m,bm),triton.cdiv(n,bn),sk)](x,w,sm,ep,base,scratch,out,m,n,k,bm,bn,bk,sk,num_warps=warps,num_stages=stages)
        if sk>1:_reduce[(triton.cdiv(m*n,256),)](scratch,out,m*n,sk,256)
    return (out,compiled) if diagnostic else out
