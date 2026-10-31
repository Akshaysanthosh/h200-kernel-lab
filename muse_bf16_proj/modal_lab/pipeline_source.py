"""Revision 3: issue, next decode and wait in one compiler-opaque PTX block."""
def fused_mma(n):
    nr=n//2
    outputs=','.join([f'"+f"(d[{i}])' for i in range(nr)]+[f'"=&r"(next[{i}])' for i in range(16)])
    inputs=','.join([f'"r"(a[{i}])' for i in range(16)]+[f'"l"(b+{2*i})' for i in range(4)]+[f'"r"(raw[{j}].{c})' for j in range(3) for c in 'xyzw']+['"r"(tab)'])
    aa=nr+16;bb=aa+16;rr=bb+4;tt=rr+12
    acc=','.join(f'%{i}' for i in range(nr))
    ops=['{ .reg .b32 rot,lo,hi;', 'wgmma.fence.sync.aligned;']
    for k in range(4):
        av=','.join(f'%{aa+4*k+i}' for i in range(4))
        ops.append(f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{acc}}}, {{{av}}}, %{bb+k}, 1, 1, 1, 0;')
    ops.append('wgmma.commit_group.sync.aligned;')
    for j in range(4):
        c=rr+8+j;l=rr+2*j;o=nr+4*j
        ops += [f'shf.r.wrap.b32 rot,%{c},%{c},4;',f'and.b32 lo,%{c},0x87878787;',f'add.u32 lo,lo,%{tt};','and.b32 hi,rot,0x87878787;',f'add.u32 hi,hi,%{tt};',f'prmt.b32 %{o},%{l},lo,0x5140;',f'prmt.b32 %{o+1},%{l},lo,0x7362;',f'prmt.b32 %{o+2},%{l+1},hi,0x5140;',f'prmt.b32 %{o+3},%{l+1},hi,0x7362;']
    ops+=['wgmma.wait_group.sync.aligned 0;','}']
    return f'D void mma{n}(float* d,uint32_t* a,uint64_t b,const uint4* raw,uint32_t tab,uint32_t* next) {{asm volatile("'+ ' '.join(ops)+f'" : {outputs} : {inputs} : "memory");}}'

def source(mod,instances,pipelined=True):
    src=mod.dd_decodediet_tt_mw_cuda_source(instances)
    marker='template<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>'
    src=src.replace(marker,'\n'.join(mod.dd_decodediet_tt_mw_mma(n) for n in (128,256))+'\n'+marker,1)
    needle='if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));'
    src=src.replace(needle,needle+'\n      if constexpr(BN==128)mma128(acc,a,desc(base+s*SB));\n      if constexpr(BN==256)mma256(acc,a,desc(base+s*SB));')
    if not pipelined:return src
    assert all(i[5] for i in instances),'Fused decoder requires compressed variants'
    for n in (16,32,64,128,256):
        original=mod.dd_decodediet_tt_mw_mma(n)
        assert original in src
        src=src.replace(original,fused_mma(n))
    begin=src.index('      uint32_t a[16];')
    end=src.index('      load(p,i+PF);',begin)
    decode=src[begin:end]
    src=src[:begin]+'      uint32_t next_a[16];\n'+src[end:]
    needle='  #pragma unroll 1\n  for(int ib=0;ib<nkb;ib+=PF) {'
    assert needle in src
    src=src.replace(needle,decode.replace('raw[p]','raw[0]')+'\n'+needle,1)
    for n in (16,32,64,128,256):
        src=src.replace(f'mma{n}(acc,a,desc(base+s*SB));',f'mma{n}(acc,a,desc(base+s*SB),raw[(p+1)%PF],t0,next_a);')
    src=src.replace('      loadx(i+Q);','      #pragma unroll\n      for(int j=0;j<16;j++)a[j]=next_a[j];\n      loadx(i+Q);')
    return src
