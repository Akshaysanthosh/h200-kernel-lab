# Original Hopper projection implementation. API usage informed by Triton 3.5
# tensor-descriptor documentation; no autotuning occurs in the submission.
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _hopper_projection(A, B, P, O, M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                       BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SK:tl.constexpr,SWAP:tl.constexpr):
    tile=tl.program_id(0)
    pn=tile % tl.cdiv(N,BN)
    pm=tile // tl.cdiv(N,BN)
    part=tl.program_id(1)
    if SWAP:
        acc=tl.zeros((BN,BM),tl.float32)
    else:
        acc=tl.zeros((BM,BN),tl.float32)
    for step in range(K//SK//BK):
        kk=part*(K//SK)+step*BK
        a=A.load([pm*BM,kk])
        b=B.load([pn*BN,kk])
        if SWAP:
            acc=tl.dot(b,tl.trans(a),acc)
        else:
            acc=tl.dot(a,tl.trans(b),acc)
    if SWAP:
        acc=tl.trans(acc)
    mm=pm*BM+tl.arange(0,BM)
    nn=pn*BN+tl.arange(0,BN)
    if SK == 1:
        tl.store(O+mm[:,None]*N+nn[None,:],acc,
                 (mm[:,None]<M)&(nn[None,:]<N))
    else:
        tl.store(P+(part*M+mm[:,None])*N+nn[None,:],acc,
                 (mm[:,None]<M)&(nn[None,:]<N))

# BM, BN, BK, split K, warps, pipeline stages. Offline selected, metadata only.
_HOPPER_SMALL=(16,128,128,4,4,3,True)
_HOPPER_LARGE=(64,128,128,2,4,3,True)
_HOPPER_SELECTED={}

def _hopper_run(data,state,config,diagnostic=False):
    x=data['x']; c=data['config']; m,k=x.shape; n=c['n']
    bm,bn,bk,sk,warps,stages,swap=config
    w=state[c['layer']]
    a=TensorDescriptor(x,[m,k],[k,1],[bm,bk])
    b=TensorDescriptor(w,[w.shape[0],k],[k,1],[bn,bk])
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    scratch=torch.empty((sk,m,n),device=x.device,dtype=torch.float32) if sk>1 else out
    compiled = _hopper_projection[(triton.cdiv(m,bm)*triton.cdiv(n,bn),sk)](
        a,b,scratch,out,m,n,k,bm,bn,bk,sk,swap,num_warps=warps,num_stages=stages)
    if sk>1:
        _reduce[(triton.cdiv(m*n,256),)](scratch,out,m*n,sk,256)
    return (out,compiled) if diagnostic else out

def kernel_fn(data,state):
    c=data['config'];x=data['x'];m=x.shape[0]
    if (c['layer'] in ('pooler','qkv_gate') and 8<=m<=1024
        and x.is_contiguous() and x.data_ptr()%16==0):
        config=_HOPPER_SELECTED.get((c['layer'],m),_HOPPER_SMALL if m<=32 else _HOPPER_LARGE)
        return _hopper_run(data,state,config)
    return _champion_kernel_fn(data,state)
