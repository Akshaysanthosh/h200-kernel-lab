"""Generate an audited one-outstanding-group decode/WGMMA pipeline. Lab only."""
def wait_helper(n):
    # Read/write constraints keep async accumulator and A registers live until
    # completion; next-fragment inputs force decoding before this wait.
    out=','.join([f'"+f"(d[{i}])' for i in range(n//2)]+[f'"+r"(live[{i}])' for i in range(16)])
    inputs=','.join(f'"r"(nxt[{i}])' for i in range(16))
    return f'D void drain{n}(float* d,uint32_t* live,const uint32_t* nxt) {{asm volatile("wgmma.wait_group.sync.aligned 0;" : {out} : {inputs} : "memory");}}'

def source(mod,instances,pipelined=True):
    src=mod.dd_decodediet_tt_mw_cuda_source(instances)
    marker='template<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>'
    helpers='\n'.join(mod.dd_decodediet_tt_mw_mma(n) for n in (128,256))
    src=src.replace(marker,helpers+'\n'+marker,1)
    needle='if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));'
    assert needle in src
    src=src.replace(needle,needle+'\n      if constexpr(BN==128)mma128(acc,a,desc(base+s*SB));\n      if constexpr(BN==256)mma256(acc,a,desc(base+s*SB));')
    if not pipelined:return src
    src=src.replace('wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;', 'wgmma.commit_group.sync.aligned;')
    # Mark register-A operands read/write at issue, making them opaque to
    # compiler copy propagation while the asynchronous instruction owns them.
    for n in (16,32,64,128,256):
        original=mod.dd_decodediet_tt_mw_mma(n).replace('wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;', 'wgmma.commit_group.sync.aligned;')
        oldouts=','.join(f'"+f"(d[{i}])' for i in range(n//2))
        oldins=','.join([f'"r"(a[{i}])' for i in range(16)]+[f'"l"(b+{2*i})' for i in range(4)])
        newouts=oldouts+','+','.join(f'"+r"(a[{i}])' for i in range(16))
        newins=','.join(f'"l"(b+{2*i})' for i in range(4))
        # Generator uses comma-space separators.
        original_out=oldouts.replace(',', ',')
        replacement=original.replace(' : '+oldouts+' : '+oldins+' : ', ' : '+newouts+' : '+newins+' : ')
        if replacement==original:
            oldouts=', '.join(f'"+f"(d[{i}])' for i in range(n//2))
            oldins=', '.join([f'"r"(a[{i}])' for i in range(16)]+[f'"l"(b+{2*i})' for i in range(4)])
            newouts=oldouts+', '+', '.join(f'"+r"(a[{i}])' for i in range(16))
            newins=', '.join(f'"l"(b+{2*i})' for i in range(4))
            replacement=original.replace(' : '+oldouts+' : '+oldins+' : ', ' : '+newouts+' : '+newins+' : ')
        assert replacement!=original
        assert original in src
        src=src.replace(original,replacement)
    helpers='\n'.join(wait_helper(n) for n in (16,32,64,128,256))
    src=src.replace(marker,helpers+'\n'+marker,1)
    src=src.replace('float acc[BN/2];','uint32_t live_a[16]={};\n  float acc[BN/2];')
    begin=src.index('      int s=i%Q;\n      waitbar(')
    end=src.index('      loadx(i+Q);',begin)+len('      loadx(i+Q);')
    drains='\n'.join(f'      if constexpr(BN=={n})drain{n}(acc,live_a,a);' for n in (16,32,64,128,256))
    issues='\n'.join(f'      if constexpr(BN=={n})mma{n}(acc,live_a,desc(base+s*SB));' for n in (16,32,64,128,256))
    src=src[:begin]+drains+'''
      // The previous group is complete: its shared slot and register A
      // fragment can now be recycled. Q>=2 keeps current/next TMA distinct.
      asm volatile("bar.sync %0, 128;" :: "r"(wg+1) : "memory");
      if(i>0)loadx(i-1+Q);
      #pragma unroll
      for(int j=0;j<16;j++)live_a[j]=a[j];
      int s=i%Q;
      waitbar(full+s*8,(i/Q)&1);
'''+issues+src[end:]
    needle='  __syncthreads();\n  if(tid%128==0) {'
    assert needle in src
    # No accumulator access or shared-memory reuse until final group drains.
    drains='\n'.join(f'  if constexpr(BN=={n})drain{n}(acc,live_a,live_a);' for n in (16,32,64,128,256))
    src=src.replace(needle,drains+'\n'+needle,1)
    assert 'loadx(i+Q);' not in src
    return src
