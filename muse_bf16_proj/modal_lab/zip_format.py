"""Lab-only grouped triple-plane lossless format; preserves v18s MMA schedule.

Original implementation informed by the TCA-TBE representation. This is not a
port of ZipServ's kernel or a reproduction of its performance claims.
"""
import numpy as np


def units(bits):
    n,k=bits.shape
    return bits.reshape(n//16,2,8,k//64,4,4,2,2).transpose(0,3,2,4,5,6,1,7).reshape(n//16,k//64,32,32)


def pack_bits(bits,cfg):
    wr,wk,ku,s=cfg
    v=units(bits)
    R,C=v.shape[:2];gx=R//wr;G=gx*s*wr*wk
    if R%wr or C!=s*wk*ku:raise ValueError('unsupported geometry')
    gw=np.arange(G)[None,:];si=np.arange(ku)[:,None]
    cta,warp=gw//(wr*wk),gw%(wr*wk)
    tile=(cta%gx)*wr+warp%wr
    chunk=(cta//gx*wk+warp//wr)*ku+si
    ordered=v[tile,chunk].reshape(-1,32,32)
    exp=((ordered>>7)&255).astype(np.int16)
    hist=np.bincount(exp.ravel(),minlength=256)
    base=int(np.convolve(hist,np.ones(7,dtype=np.int64),'valid').argmax())-1
    common=(exp>base)&(exp<=base+7)
    codes=np.where(common,exp-base,0).astype(np.uint32)
    shifts=np.arange(32,dtype=np.uint32)[None,None,:]
    planes=np.stack([(((codes>>b)&1)<<shifts).sum(-1,dtype=np.uint32) for b in range(3)],axis=1)
    sm=((ordered&127)|((ordered>>8)&128)).astype(np.uint8)
    nunit=ordered.shape[0];counts=common.sum((1,2));padcommon=(counts+15)//16*16;padfull=(2*(1024-counts)+15)//16*16
    lengths=384+padcommon+padfull
    offsets=np.r_[0,np.cumsum(lengths)].astype(np.uint32)
    meta=np.stack([offsets[:-1],offsets[:-1]+384+padcommon],axis=1).astype(np.uint32)
    packed=np.zeros(int(offsets[-1]),dtype=np.uint8)
    for u in range(nunit):
        a=int(offsets[u]);f=int(meta[u,1]);cm=common[u]
        packed[a:a+384]=planes[u].ravel().view(np.uint8)
        packed[a+384:a+384+int(counts[u])]=sm[u][cm]
        full=ordered[u][~cm].astype('<u2').view(np.uint8)
        packed[f:f+full.size]=full
    return {'packed':packed,'meta':meta,'base':base,'ordered':ordered,
            'bpw':(packed.nbytes+meta.nbytes+4)*8/bits.size,'exception_rate':float(1-common.mean())}


def unpack(p):
    data,meta,base=p['packed'],p['meta'],p['base']
    out=np.empty((len(meta),32,32),dtype=np.uint16)
    for u,(a,f) in enumerate(meta.astype(np.int64)):
        planes=data[a:a+384].view('<u4').reshape(3,32)
        codes=sum((((planes[b,:,None]>>np.arange(32,dtype=np.uint32))&1)<<b) for b in range(3))
        common=codes!=0;nc=int(common.sum())
        sm=data[a+384:a+384+nc].astype(np.uint16)
        out[u][common]=(sm&127)|((sm&128)<<8)|((codes[common].astype(np.int32)+base).astype(np.uint16)<<7)
        out[u][~common]=data[f:f+2*(1024-nc)].view('<u2')
    return out


LOADER=r'''
SM3D void zip_load(const int4* raw, const uint32_t* meta, int unit, int lane,
                  int base, int4 (&w)[4]) {
  const uint8_t* data=(const uint8_t*)raw;
  const uint32_t a=meta[2*unit], f=meta[2*unit+1];
  const uint32_t* p=(const uint32_t*)(data+a);
  uint32_t p0=v18s_lde(p+lane),p1=v18s_lde(p+32+lane),p2=v18s_lde(p+64+lane);
  uint32_t mask=p0|p1|p2;
  int count=__popc(mask),prefix=count;
#pragma unroll
  for(int d=1;d<32;d*=2){int z=__shfl_up_sync(0xffffffffu,prefix,d);if(lane>=d)prefix+=z;}
  prefix-=count;
  const uint8_t* common=data+a+384+prefix;
  const uint16_t* full=(const uint16_t*)(data+f)+lane*32-prefix;
  uint32_t vals[16];
#pragma unroll
  for(int q=0;q<16;++q){
    uint32_t pair=0;
#pragma unroll
    for(int h=0;h<2;++h){
      int i=2*q+h;
      uint32_t code=((p0>>i)&1)|(((p1>>i)&1)<<1)|(((p2>>i)&1)<<2);
      uint32_t before= i==0 ? 0u : (mask & (0xffffffffu>>(32-i)));
      int ci=__popc(before);
      uint32_t v;
      if(code){uint32_t sm=common[ci];v=(sm&127u)|((sm&128u)<<8)|((base+code)<<7);}
      else v=full[i-ci];
      pair|=v<<(16*h);
    }
    vals[q]=pair;
  }
#pragma unroll
  for(int j=0;j<4;++j)w[j]=make_int4(vals[4*j],vals[4*j+1],vals[4*j+2],vals[4*j+3]);
}
'''


def source(mod,instances):
    src=mod.v18s_cuda_source(instances)
    needle='template <int FMT> struct Sm3Fmt;'
    src=src.replace(needle,LOADER+'\n'+needle)
    old='for (int l = 0; l < NL; ++l) w[slot][j][l] = v18s_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);'
    # RT=1, FMT=0 only, using the exact same stream unit order as v18s_stream.
    if src.count(old)!=1:raise RuntimeError('loader changed upstream')
    src=src.replace('#pragma unroll\n        '+old,'        zip_load(wp, esc, (s * G + gw) * RT + j, lane, T0, w[slot][j]);')
    if old in src:raise RuntimeError('loader replacement failed')
    return src


def audit():
    # Every BF16 bit pattern, including signed zeros, subnormals, inf/NaN payloads.
    bits=np.arange(65536,dtype=np.uint16).reshape(128,512)
    p=pack_bits(bits,(1,2,4,1))
    assert np.array_equal(unpack(p),p['ordered'])
    return {'all_bf16_patterns':65536,'roundtrip':True}


if __name__=='__main__':print(audit())
