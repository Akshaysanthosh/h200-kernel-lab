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
