"""Lab-only router: one BF16 dot over interleaved hi/lo K, FP32 reduction."""
import triton
import triton.language as tl

@triton.jit
def interleave_dot(X,W,P,M:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SK:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM)
    ni=tl.program_id(1)*BN+tl.arange(0,BN)
    si=tl.program_id(2)
    ki=si*(16384//SK)+tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for block in range(16384//SK//BK):
        kk=ki+block*BK
        a=tl.load(X+mi[:,None]*8192+kk[None,:]//2,mi[:,None]<M,other=0)
        b=tl.load(W+kk[:,None]*256+ni[None,:])
        acc=tl.dot(a,b,acc)
    tl.store(P+si*M*256+mi[:,None]*256+ni[None,:],acc,mi[:,None]<M)

@triton.jit
def reduce_parts(P,Y,M:tl.constexpr,SK:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    s=tl.arange(0,SK)
    v=tl.load(P+s[:,None]*M*256+i[None,:],i[None,:]<M*256,other=0)
    tl.store(Y+i,tl.sum(v,axis=0),i<M*256)

def pack(w):
    import torch
    hi=w.to(torch.bfloat16);lo=(w-hi.float()).to(torch.bfloat16)
    return torch.stack((hi.T,lo.T),dim=1).contiguous().reshape(16384,256)

def run(x,w,p,y,cfg):
    bm,bn,bk,sk,warps,stages=cfg
    interleave_dot[(triton.cdiv(x.shape[0],bm),256//bn,sk)](x,w,p,x.shape[0],bm,bn,bk,sk,num_warps=warps,num_stages=stages)
    if sk==1:return p[0]
    reduce_parts[(triton.cdiv(x.shape[0]*256,256),)](p,y,x.shape[0],sk,256)
    return y
