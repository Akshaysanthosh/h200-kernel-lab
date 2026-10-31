# Original direct-load projections. Fixed offline configurations only.
@triton.jit
def _other_dot(X,W,L,P,O,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
               BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SK:tl.constexpr,
               ROUTER:tl.constexpr,SWAP:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM)
    ni=tl.program_id(1)*BN+tl.arange(0,BN)
    part=tl.program_id(2)
    kk=part*(K//SK)+tl.arange(0,BK)
    if SWAP: acc=tl.zeros((BN,BM),tl.float32)
    else: acc=tl.zeros((BM,BN),tl.float32)
    if ROUTER:
        if SWAP: lowacc=tl.zeros((BN,BM),tl.float32)
        else: lowacc=tl.zeros((BM,BN),tl.float32)
    for step in range(K//SK//BK):
        ki=kk+step*BK
        x=tl.load(X+mi[:,None]*K+ki[None,:],mi[:,None]<M,0)
        w=tl.load(W+ni[:,None]*K+ki[None,:],ni[:,None]<N,0)
        if SWAP: acc=tl.dot(w,tl.trans(x),acc)
        else: acc=tl.dot(x,tl.trans(w),acc)
        if ROUTER:
            lo=tl.load(L+ni[:,None]*K+ki[None,:],ni[:,None]<N,0)
            if SWAP: lowacc=tl.dot(lo,tl.trans(x),lowacc)
            else: lowacc=tl.dot(x,tl.trans(lo),lowacc)
    if ROUTER: acc=acc+lowacc
    if SWAP: acc=tl.trans(acc)
    offset=mi[:,None]*N+ni[None,:]
    if SK==1: tl.store(O+offset,acc,(mi[:,None]<M)&(ni[None,:]<N))
    else: tl.store(P+part*M*N+offset,acc,(mi[:,None]<M)&(ni[None,:]<N))

@triton.jit
def _other_vector(X,W,O,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    # Each CTA owns complete output elements: no global reduction or atomics.
    row=tl.program_id(0)*R+tl.arange(0,R)
    token=tl.program_id(1)
    kk=tl.arange(0,B)
    x=tl.load(X+token*K+kk,kk<K,0).to(tl.float32)
    w=tl.load(W+row[:,None]*K+kk[None,:],(row[:,None]<N)&(kk[None,:]<K),0).to(tl.float32)
    val=tl.sum(w*x[None,:],1)
    tl.store(O+token*N+row,val,row<N)

def _other_run(data,state,config,diagnostic=False):
    x=data['x']; c=data['config']; m,k=x.shape; n=c['n']; router=c['layer']=='router'
    out=torch.empty((m,n),device=x.device,dtype=torch.float32 if router else torch.bfloat16)
    if config[0]=='vector':
        _,rows,warps=config
        compiled=_other_vector[(triton.cdiv(n,rows),m)](x,state[c['layer']],out,m,n,k,rows,triton.next_power_of_2(k),num_warps=warps)
    else:
        _,bm,bn,bk,sk,warps,stages,swap=config
        scratch=torch.empty((sk,m,n),device=x.device,dtype=torch.float32) if sk>1 else out
        compiled=_other_dot[(triton.cdiv(m,bm),triton.cdiv(n,bn),sk)](x,state['hi'] if router else state[c['layer']],state['lo'] if router else state[c['layer']],scratch,out,m,n,k,bm,bn,bk,sk,router,swap,num_warps=warps,num_stages=stages)
        if sk>1:_reduce[(triton.cdiv(m*n,256),)](scratch,out,m*n,sk,256)
    return (out,compiled) if diagnostic else out
