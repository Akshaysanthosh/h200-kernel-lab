// dd_cap2_b_


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define D __device__ __forceinline__
D uint32_t addr(void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
D void waitbar(uint32_t b,int p) {
  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");
}
D void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
D void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
D uint64_t desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }
D void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");
}
D uint4 ldw(const uint4* p,uint64_t policy) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3},[%4],%5;" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p),"l"(policy));
  return v;
}
D uint32_t prmt(uint32_t a,uint32_t b,uint32_t s) {
  uint32_t d;asm("prmt.b32 %0,%1,%2,%3;":"=r"(d):"r"(a),"r"(b),"r"(s));return d;
}
D void dec(uint32_t L0,uint32_t L1,uint32_t C,uint32_t T0,uint32_t T1,uint32_t* a) {
  uint32_t rotated;
  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));
  const uint32_t low_high = (C & 0x87878787u) + T0;
  const uint32_t high_high = (rotated & 0x87878787u) + T0;
  a[0] = prmt(L0, low_high, 0x5140u);
  a[1] = prmt(L0, low_high, 0x7362u);
  a[2] = prmt(L1, high_high, 0x5140u);
  a[3] = prmt(L1, high_high, 0x7362u);
}
D void patch(uint32_t* a,uint32_t e) {
  uint32_t reg=(e>>8)&31u,xr=(e&127u)<<(8+((e>>7)&1u)*16);
  #pragma unroll
  for(int r=0;r<16;r++) if(reg==r)a[r]^=xr;
}

D void mma16(float* d,uint32_t* a,uint64_t b) {asm volatile("wgmma.fence.sync.aligned; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %24, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%12,%13,%14,%15}, %25, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%16,%17,%18,%19}, %26, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%20,%21,%22,%23}, %27, 1, 1, 1, 0; wgmma.commit_group.sync.aligned;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+r"(a[0]),"+r"(a[1]),"+r"(a[2]),"+r"(a[3]),"+r"(a[4]),"+r"(a[5]),"+r"(a[6]),"+r"(a[7]),"+r"(a[8]),"+r"(a[9]),"+r"(a[10]),"+r"(a[11]),"+r"(a[12]),"+r"(a[13]),"+r"(a[14]),"+r"(a[15]) : "l"(b+0),"l"(b+2),"l"(b+4),"l"(b+6) : "memory");}
D void mma32(float* d,uint32_t* a,uint64_t b) {asm volatile("wgmma.fence.sync.aligned; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %32, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%20,%21,%22,%23}, %33, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%24,%25,%26,%27}, %34, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%28,%29,%30,%31}, %35, 1, 1, 1, 0; wgmma.commit_group.sync.aligned;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+r"(a[0]),"+r"(a[1]),"+r"(a[2]),"+r"(a[3]),"+r"(a[4]),"+r"(a[5]),"+r"(a[6]),"+r"(a[7]),"+r"(a[8]),"+r"(a[9]),"+r"(a[10]),"+r"(a[11]),"+r"(a[12]),"+r"(a[13]),"+r"(a[14]),"+r"(a[15]) : "l"(b+0),"l"(b+2),"l"(b+4),"l"(b+6) : "memory");}
D void mma64(float* d,uint32_t* a,uint64_t b) {asm volatile("wgmma.fence.sync.aligned; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %48, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%36,%37,%38,%39}, %49, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%40,%41,%42,%43}, %50, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%44,%45,%46,%47}, %51, 1, 1, 1, 0; wgmma.commit_group.sync.aligned;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+r"(a[0]),"+r"(a[1]),"+r"(a[2]),"+r"(a[3]),"+r"(a[4]),"+r"(a[5]),"+r"(a[6]),"+r"(a[7]),"+r"(a[8]),"+r"(a[9]),"+r"(a[10]),"+r"(a[11]),"+r"(a[12]),"+r"(a[13]),"+r"(a[14]),"+r"(a[15]) : "l"(b+0),"l"(b+2),"l"(b+4),"l"(b+6) : "memory");}

D void mma128(float* d,uint32_t* a,uint64_t b) {asm volatile("wgmma.fence.sync.aligned; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %80, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%68,%69,%70,%71}, %81, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%72,%73,%74,%75}, %82, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%76,%77,%78,%79}, %83, 1, 1, 1, 0; wgmma.commit_group.sync.aligned;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),"+r"(a[0]),"+r"(a[1]),"+r"(a[2]),"+r"(a[3]),"+r"(a[4]),"+r"(a[5]),"+r"(a[6]),"+r"(a[7]),"+r"(a[8]),"+r"(a[9]),"+r"(a[10]),"+r"(a[11]),"+r"(a[12]),"+r"(a[13]),"+r"(a[14]),"+r"(a[15]) : "l"(b+0),"l"(b+2),"l"(b+4),"l"(b+6) : "memory");}
D void mma256(float* d,uint32_t* a,uint64_t b) {asm volatile("wgmma.fence.sync.aligned; wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, {%128,%129,%130,%131}, %144, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, {%132,%133,%134,%135}, %145, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, {%136,%137,%138,%139}, %146, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, {%140,%141,%142,%143}, %147, 1, 1, 1, 0; wgmma.commit_group.sync.aligned;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),"+f"(d[64]),"+f"(d[65]),"+f"(d[66]),"+f"(d[67]),"+f"(d[68]),"+f"(d[69]),"+f"(d[70]),"+f"(d[71]),"+f"(d[72]),"+f"(d[73]),"+f"(d[74]),"+f"(d[75]),"+f"(d[76]),"+f"(d[77]),"+f"(d[78]),"+f"(d[79]),"+f"(d[80]),"+f"(d[81]),"+f"(d[82]),"+f"(d[83]),"+f"(d[84]),"+f"(d[85]),"+f"(d[86]),"+f"(d[87]),"+f"(d[88]),"+f"(d[89]),"+f"(d[90]),"+f"(d[91]),"+f"(d[92]),"+f"(d[93]),"+f"(d[94]),"+f"(d[95]),"+f"(d[96]),"+f"(d[97]),"+f"(d[98]),"+f"(d[99]),"+f"(d[100]),"+f"(d[101]),"+f"(d[102]),"+f"(d[103]),"+f"(d[104]),"+f"(d[105]),"+f"(d[106]),"+f"(d[107]),"+f"(d[108]),"+f"(d[109]),"+f"(d[110]),"+f"(d[111]),"+f"(d[112]),"+f"(d[113]),"+f"(d[114]),"+f"(d[115]),"+f"(d[116]),"+f"(d[117]),"+f"(d[118]),"+f"(d[119]),"+f"(d[120]),"+f"(d[121]),"+f"(d[122]),"+f"(d[123]),"+f"(d[124]),"+f"(d[125]),"+f"(d[126]),"+f"(d[127]),"+r"(a[0]),"+r"(a[1]),"+r"(a[2]),"+r"(a[3]),"+r"(a[4]),"+r"(a[5]),"+r"(a[6]),"+r"(a[7]),"+r"(a[8]),"+r"(a[9]),"+r"(a[10]),"+r"(a[11]),"+r"(a[12]),"+r"(a[13]),"+r"(a[14]),"+r"(a[15]) : "l"(b+0),"l"(b+2),"l"(b+4),"l"(b+6) : "memory");}
D void drain16(float* d,uint32_t* live,const uint32_t* nxt) {asm volatile("wgmma.wait_group.sync.aligned 0;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+r"(live[0]),"+r"(live[1]),"+r"(live[2]),"+r"(live[3]),"+r"(live[4]),"+r"(live[5]),"+r"(live[6]),"+r"(live[7]),"+r"(live[8]),"+r"(live[9]),"+r"(live[10]),"+r"(live[11]),"+r"(live[12]),"+r"(live[13]),"+r"(live[14]),"+r"(live[15]) : "r"(nxt[0]),"r"(nxt[1]),"r"(nxt[2]),"r"(nxt[3]),"r"(nxt[4]),"r"(nxt[5]),"r"(nxt[6]),"r"(nxt[7]),"r"(nxt[8]),"r"(nxt[9]),"r"(nxt[10]),"r"(nxt[11]),"r"(nxt[12]),"r"(nxt[13]),"r"(nxt[14]),"r"(nxt[15]) : "memory");}
D void drain32(float* d,uint32_t* live,const uint32_t* nxt) {asm volatile("wgmma.wait_group.sync.aligned 0;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+r"(live[0]),"+r"(live[1]),"+r"(live[2]),"+r"(live[3]),"+r"(live[4]),"+r"(live[5]),"+r"(live[6]),"+r"(live[7]),"+r"(live[8]),"+r"(live[9]),"+r"(live[10]),"+r"(live[11]),"+r"(live[12]),"+r"(live[13]),"+r"(live[14]),"+r"(live[15]) : "r"(nxt[0]),"r"(nxt[1]),"r"(nxt[2]),"r"(nxt[3]),"r"(nxt[4]),"r"(nxt[5]),"r"(nxt[6]),"r"(nxt[7]),"r"(nxt[8]),"r"(nxt[9]),"r"(nxt[10]),"r"(nxt[11]),"r"(nxt[12]),"r"(nxt[13]),"r"(nxt[14]),"r"(nxt[15]) : "memory");}
D void drain64(float* d,uint32_t* live,const uint32_t* nxt) {asm volatile("wgmma.wait_group.sync.aligned 0;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+r"(live[0]),"+r"(live[1]),"+r"(live[2]),"+r"(live[3]),"+r"(live[4]),"+r"(live[5]),"+r"(live[6]),"+r"(live[7]),"+r"(live[8]),"+r"(live[9]),"+r"(live[10]),"+r"(live[11]),"+r"(live[12]),"+r"(live[13]),"+r"(live[14]),"+r"(live[15]) : "r"(nxt[0]),"r"(nxt[1]),"r"(nxt[2]),"r"(nxt[3]),"r"(nxt[4]),"r"(nxt[5]),"r"(nxt[6]),"r"(nxt[7]),"r"(nxt[8]),"r"(nxt[9]),"r"(nxt[10]),"r"(nxt[11]),"r"(nxt[12]),"r"(nxt[13]),"r"(nxt[14]),"r"(nxt[15]) : "memory");}
D void drain128(float* d,uint32_t* live,const uint32_t* nxt) {asm volatile("wgmma.wait_group.sync.aligned 0;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),"+r"(live[0]),"+r"(live[1]),"+r"(live[2]),"+r"(live[3]),"+r"(live[4]),"+r"(live[5]),"+r"(live[6]),"+r"(live[7]),"+r"(live[8]),"+r"(live[9]),"+r"(live[10]),"+r"(live[11]),"+r"(live[12]),"+r"(live[13]),"+r"(live[14]),"+r"(live[15]) : "r"(nxt[0]),"r"(nxt[1]),"r"(nxt[2]),"r"(nxt[3]),"r"(nxt[4]),"r"(nxt[5]),"r"(nxt[6]),"r"(nxt[7]),"r"(nxt[8]),"r"(nxt[9]),"r"(nxt[10]),"r"(nxt[11]),"r"(nxt[12]),"r"(nxt[13]),"r"(nxt[14]),"r"(nxt[15]) : "memory");}
D void drain256(float* d,uint32_t* live,const uint32_t* nxt) {asm volatile("wgmma.wait_group.sync.aligned 0;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),"+f"(d[64]),"+f"(d[65]),"+f"(d[66]),"+f"(d[67]),"+f"(d[68]),"+f"(d[69]),"+f"(d[70]),"+f"(d[71]),"+f"(d[72]),"+f"(d[73]),"+f"(d[74]),"+f"(d[75]),"+f"(d[76]),"+f"(d[77]),"+f"(d[78]),"+f"(d[79]),"+f"(d[80]),"+f"(d[81]),"+f"(d[82]),"+f"(d[83]),"+f"(d[84]),"+f"(d[85]),"+f"(d[86]),"+f"(d[87]),"+f"(d[88]),"+f"(d[89]),"+f"(d[90]),"+f"(d[91]),"+f"(d[92]),"+f"(d[93]),"+f"(d[94]),"+f"(d[95]),"+f"(d[96]),"+f"(d[97]),"+f"(d[98]),"+f"(d[99]),"+f"(d[100]),"+f"(d[101]),"+f"(d[102]),"+f"(d[103]),"+f"(d[104]),"+f"(d[105]),"+f"(d[106]),"+f"(d[107]),"+f"(d[108]),"+f"(d[109]),"+f"(d[110]),"+f"(d[111]),"+f"(d[112]),"+f"(d[113]),"+f"(d[114]),"+f"(d[115]),"+f"(d[116]),"+f"(d[117]),"+f"(d[118]),"+f"(d[119]),"+f"(d[120]),"+f"(d[121]),"+f"(d[122]),"+f"(d[123]),"+f"(d[124]),"+f"(d[125]),"+f"(d[126]),"+f"(d[127]),"+r"(live[0]),"+r"(live[1]),"+r"(live[2]),"+r"(live[3]),"+r"(live[4]),"+r"(live[5]),"+r"(live[6]),"+r"(live[7]),"+r"(live[8]),"+r"(live[9]),"+r"(live[10]),"+r"(live[11]),"+r"(live[12]),"+r"(live[13]),"+r"(live[14]),"+r"(live[15]) : "r"(nxt[0]),"r"(nxt[1]),"r"(nxt[2]),"r"(nxt[3]),"r"(nxt[4]),"r"(nxt[5]),"r"(nxt[6]),"r"(nxt[7]),"r"(nxt[8]),"r"(nxt[9]),"r"(nxt[10]),"r"(nxt[11]),"r"(nxt[12]),"r"(nxt[13]),"r"(nxt[14]),"r"(nxt[15]) : "memory");}
template<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>
__global__ __launch_bounds__(128*WK,CAP) void m32w_gemm(
    const __grid_constant__ CUtensorMap tx,const uint4* __restrict__ w,
    const uint32_t* __restrict__ esc,__nv_bfloat16* out,int m,int n,int k,int split,uint32_t t0,uint32_t t1) {
  extern __shared__ __align__(1024) uint8_t sm[];
  constexpr int RW=R/16,NT=128*WK,RP=R+4,SB=BN*128,WS=(Q*(SB+8)+1023)/1024*1024,NL=C12?3:4;
  int tid=threadIdx.x,wg=tid/128,warp=tid%128/32,lane=tid%32;
  uint32_t base0=addr(sm),base=base0+wg*WS,full=base+Q*SB;
  int rank=blockIdx.x%split,nt=blockIdx.x/split,mt=blockIdx.y;
  int kb0=(k/64)*(rank*WK+wg)/(split*WK),nkb=(k/64)*(rank*WK+wg+1)/(split*WK)-kb0;
  uint64_t pw,px;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0,1.0;":"=l"(pw));
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0,1.0;":"=l"(px));
  if(tid%128==0) {
    for(int s=0;s<Q;s++)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  auto loadx=[&](int i) {
    if(tid%128==0 && i<nkb) {
      int s=i%Q;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");
      tma(base+s*SB,&tx,(kb0+i)*64,mt*BN,full+s*8,px);
    }
  };
  #pragma unroll
  for(int s=0;s<Q;s++)loadx(s);
  uint32_t e0=0xffffffffu,e1=0xffffffffu,nxt=0xffffffffu;int ep=0;
  if(C12 && warp<RW) {
    const uint32_t* el=esc+((nt*split*WK+rank*WK+wg)*RW+warp)*64;
    e0=el[lane];e1=el[lane+32];nxt=__shfl_sync(0xffffffffu,e0,0);
  }
  uint4 raw[PF][NL]={};
  auto load=[&](int slot,int i) {
    if(i<nkb && warp<RW) {
      const uint4* src=w+((long)(nt*(k/64)+kb0+i)*RW+warp)*NL*32+lane;
      #pragma unroll
      for(int l=0;l<NL;l++)raw[slot][l]=ldw(src+l*32,pw);
    }
  };
  #pragma unroll
  for(int p=0;p<PF;p++)load(p,p);
  uint32_t live_a[16]={};
  float acc[BN/2];
  #pragma unroll
  for(int j=0;j<BN/2;j++)asm volatile("mov.b32 %0,0;":"=f"(acc[j]));
  #pragma unroll 1
  for(int ib=0;ib<nkb;ib+=PF) {
    #pragma unroll
    for(int p=0;p<PF;p++) {
      int i=ib+p;if(i>=nkb)break;
      uint32_t a[16];
      if constexpr(C12) {
        dec(raw[p][0].x,raw[p][0].y,raw[p][2].x,t0,t1,a);
        dec(raw[p][0].z,raw[p][0].w,raw[p][2].y,t0,t1,a+4);
        dec(raw[p][1].x,raw[p][1].y,raw[p][2].z,t0,t1,a+8);
        dec(raw[p][1].z,raw[p][1].w,raw[p][2].w,t0,t1,a+12);
      } else {
        #pragma unroll
        for(int l=0;l<4;l++) {
          a[4*l]=raw[p][l].x;a[4*l+1]=raw[p][l].y;a[4*l+2]=raw[p][l].z;a[4*l+3]=raw[p][l].w;
        }
      }
      load(p,i+PF);
      if constexpr(C12) {
        while((nxt>>18)==uint32_t(i)) {
          if(lane==int((nxt>>13)&31u))patch(a,nxt);
          ++ep;
          uint32_t ev=ep<32?e0:e1;
          nxt=__shfl_sync(0xffffffffu,ev,ep&31);
          if(ep>=64)nxt=0xffffffffu;
        }
      }
      if constexpr(BN==16)drain16(acc,live_a,a);
      if constexpr(BN==32)drain32(acc,live_a,a);
      if constexpr(BN==64)drain64(acc,live_a,a);
      if constexpr(BN==128)drain128(acc,live_a,a);
      if constexpr(BN==256)drain256(acc,live_a,a);
      // The previous group is complete: its shared slot and register A
      // fragment can now be recycled. Q>=2 keeps current/next TMA distinct.
      asm volatile("bar.sync %0, 128;" :: "r"(wg+1) : "memory");
      if(i>0)loadx(i-1+Q);
      #pragma unroll
      for(int j=0;j<16;j++)live_a[j]=a[j];
      int s=i%Q;
      waitbar(full+s*8,(i/Q)&1);
      if constexpr(BN==16)mma16(acc,live_a,desc(base+s*SB));
      if constexpr(BN==32)mma32(acc,live_a,desc(base+s*SB));
      if constexpr(BN==64)mma64(acc,live_a,desc(base+s*SB));
      if constexpr(BN==128)mma128(acc,live_a,desc(base+s*SB));
      if constexpr(BN==256)mma256(acc,live_a,desc(base+s*SB));
    }
  }
  if constexpr(BN==16)drain16(acc,live_a,live_a);
  if constexpr(BN==32)drain32(acc,live_a,live_a);
  if constexpr(BN==64)drain64(acc,live_a,live_a);
  if constexpr(BN==128)drain128(acc,live_a,live_a);
  if constexpr(BN==256)drain256(acc,live_a,live_a);
  __syncthreads();
  if(tid%128==0) {
    for(int s=0;s<Q;s++)asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(full+s*8) : "memory");
  }
  __syncthreads();
  int row=warp*16+lane/4,col=2*(lane%4);
  #pragma unroll
  for(int j=0;j<BN/8;j++) {
    #pragma unroll
    for(int h=0;h<2;h++) {
      if(warp>=RW)continue;
      ((float*)sm)[(wg*BN+col+8*j)*RP+row+8*h]=acc[4*j+2*h];
      ((float*)sm)[(wg*BN+col+8*j+1)*RP+row+8*h]=acc[4*j+2*h+1];
    }
  }
  __syncthreads();
  if(split>1)cluster_sync();
  for(int q=tid+rank*NT;q<BN*(R/4);q+=NT*split) {
    int token=q/(R/4),row=(q%(R/4))*4;
    if(mt*BN+token>=m)continue;
    float4 sum=make_float4(0,0,0,0);
    for(int pp=0;pp<split*WK;pp++) {
      int s=pp/WK;
      uint32_t remote=base0+((pp%WK*BN+token)*RP+row)*4;
      if(split>1)asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(remote):"r"(remote),"r"(s));
      float4 v;
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(remote):"memory");
      sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;
    }
    __nv_bfloat162 lo=__floats2bfloat162_rn(sum.x,sum.y),hi=__floats2bfloat162_rn(sum.z,sum.w);
    *(uint2*)(out+(long)(mt*BN+token)*n+nt*R+row)=make_uint2(*(uint32_t*)&lo,*(uint32_t*)&hi);
  }
  __syncthreads();
  if(split>1)cluster_sync();
}


using EncodeFn=decltype(&cuTensorMapEncodeTiled);
int map(CUtensorMap* tm,const void* ptr,int rows,int k,int box) {
  static EncodeFn encode=nullptr;
  if(!encode) {cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);}
  uint64_t dims[]={uint64_t(k),uint64_t(rows)},stride[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}


template<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>
int launch_t(const void* x,const void* w,const void* e,void* out,int m,int n,int k,int split,uint64_t tab,cudaStream_t st) {
  CUtensorMap tx;int rc=map(&tx,x,m,k,BN);if(rc)return rc;
  constexpr int a=WK*((Q*(BN*128+8)+1023)/1024*1024),b=WK*BN*(R+4)*4,smem=a>b?a:b;
  auto kern=m32w_gemm<BN,R,WK,PF,Q,C12,CAP>;
  static bool ready=false;
  if(!ready) {
    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));if(rc)return rc;
    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));if(rc)return rc;
    ready=true;
  }
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((n/R)*split,(m+BN-1)/BN);cfg.blockDim=dim3(128*WK);cfg.dynamicSmemBytes=smem;cfg.stream=st;
  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={unsigned(split),1,1};
  if(split>1){cfg.attrs=&attr;cfg.numAttrs=1;}
  return int(cudaLaunchKernelEx(&cfg,kern,tx,(const uint4*)w,(const uint32_t*)e,(__nv_bfloat16*)out,m,n,k,split,uint32_t(tab),uint32_t(tab>>32)));
}
int launch(int id,uint64_t x,uint64_t w,uint64_t e,uint64_t out,int m,int n,int k,int split,uint64_t tab,uint64_t st) {
  switch(id) {case 0: return launch_t<64,64,1,1,2,true,1>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);
case 1: return launch_t<64,64,1,2,2,true,1>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);
case 2: return launch_t<64,64,2,1,2,true,1>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);
case 3: return launch_t<64,64,2,2,2,true,1>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);
case 4: return launch_t<128,64,1,1,2,true,1>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);
case 5: return launch_t<128,64,1,2,2,true,1>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);
case 6: return launch_t<128,64,2,1,2,true,1>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);
case 7: return launch_t<128,64,2,2,2,true,1>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);
case 8: return launch_t<256,64,1,1,2,true,1>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);
case 9: return launch_t<256,64,1,2,2,true,1>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);
case 10: return launch_t<256,64,2,1,2,true,1>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);
case 11: return launch_t<256,64,2,2,2,true,1>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);}
  return -1;
}
int info(int id){cudaFuncAttributes fa={};switch(id){case 0:cudaFuncGetAttributes(&fa,m32w_gemm<64,64,1,1,2,true,1>);break;
case 1:cudaFuncGetAttributes(&fa,m32w_gemm<64,64,1,2,2,true,1>);break;
case 2:cudaFuncGetAttributes(&fa,m32w_gemm<64,64,2,1,2,true,1>);break;
case 3:cudaFuncGetAttributes(&fa,m32w_gemm<64,64,2,2,2,true,1>);break;
case 4:cudaFuncGetAttributes(&fa,m32w_gemm<128,64,1,1,2,true,1>);break;
case 5:cudaFuncGetAttributes(&fa,m32w_gemm<128,64,1,2,2,true,1>);break;
case 6:cudaFuncGetAttributes(&fa,m32w_gemm<128,64,2,1,2,true,1>);break;
case 7:cudaFuncGetAttributes(&fa,m32w_gemm<128,64,2,2,2,true,1>);break;
case 8:cudaFuncGetAttributes(&fa,m32w_gemm<256,64,1,1,2,true,1>);break;
case 9:cudaFuncGetAttributes(&fa,m32w_gemm<256,64,1,2,2,true,1>);break;
case 10:cudaFuncGetAttributes(&fa,m32w_gemm<256,64,2,1,2,true,1>);break;
case 11:cudaFuncGetAttributes(&fa,m32w_gemm<256,64,2,2,2,true,1>);break;}return fa.numRegs*10000+fa.localSizeBytes;}
