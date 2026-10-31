"""Same packed weights and exceptions; replace divergent register repair with PTX predication."""
import re


def source(mod,instances):
    src=mod.v18s_cuda_source(instances)
    blocks=[]
    for nr in (16,32):
        lines=['{ .reg .pred owner, selected;',f'setp.eq.u32 owner, %{nr+2}, %{nr+3};']
        for r in range(nr):
            lines.extend([f'setp.eq.and.u32 selected, %{nr}, {r}, owner;',f'@selected xor.b32 %{r}, %{r}, %{nr+1};'])
        lines.append('}')
        quoted='\n'.join('      "'+s+'\\n"' for s in lines)
        outputs=', '.join(f'"+r"(A[{r}])' for r in range(nr))
        blocks.append(f'''  if constexpr (NR == {nr}) {{
    asm volatile(
{quoted}
      : {outputs}
      : "r"(reg), "r"(xr), "r"(lane), "r"(owner));
  }}''')
    helper='''template <int NR>
SM3D void v18s_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
  const uint32_t owner = (e >> 13) & 31u;
  const uint32_t lane = threadIdx.x & 31;
'''+ '\n'.join(blocks)+'\n}'
    src,n=re.subn(r'template <int NR>\nSM3D void v18s_patch\([^\n]+\) \{.*?\n\}',lambda _:helper,src,flags=re.S)
    assert n==1
    needle='if (lane == (int)((nxt >> 13) & 31u)) v18s_patch<NA * 16>(A[j], nxt);'
    assert src.count(needle)==1
    return src.replace(needle,'v18s_patch<NA * 16>(A[j], nxt);')


def audit():
    cases=0
    for nr in (16,32):
      for owner in range(32):
       for reg in range(nr):
        for half in range(2):
         for xor in (0,1,63,127):
          entry=(owner<<13)|(reg<<8)|(half<<7)|xor
          for lane in range(32):
           for r in range(nr):
            expected=(xor<<(8+16*half)) if lane==owner and r==reg else 0
            selected=(lane==((entry>>13)&31) and r==((entry>>8)&31))
            got=((entry&127)<<(8+((entry>>7)&1)*16)) if selected else 0
            assert got==expected
          cases+=1
    return {'exception_patterns':cases,'passed':True}

if __name__=='__main__':print(audit())
