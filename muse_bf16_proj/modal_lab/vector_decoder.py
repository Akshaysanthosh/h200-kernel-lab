"""Lab-only aligned-prefix decoder: same vector loads, fewer common-path ALU ops."""
import re,inspect
from decode_variants import source as arithmetic_source


def source(mod,instances,cap=64):
    src=arithmetic_source(mod,instances)
    src=src.replace('(C & 0x87878787u) + T0','(C & 0x87878787u) | T0').replace('(rotated & 0x87878787u) + T0','(rotated & 0x87878787u) | T0')
    src=src.replace('esc + (size_t)gw * 32 + lane',f'esc + (size_t)gw * {cap} + lane')
    old='nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;'
    new=f'''if ((ep & 31) == 0 && ep < {cap}) my_e = v18s_lde(esc + (size_t)gw * {cap} + ep + lane);
                  nxt = ep < {cap} ? __shfl_sync(0xffffffffu, my_e, ep & 31) : 0xFFFFFFFFu;'''
    assert src.count(old)==1
    return src.replace(old,new)


def pack(mod,w):
    import torch
    v=mod.v18s_units(w);r,c=v.shape[:2];dev=v.device
    lo=v&255;sign=v>>15;eh=(v>>8)&127
    hist=torch.bincount(eh.flatten(),minlength=128).cpu()
    bases=list(range(0,121,8));base=bases[max(range(len(bases)),key=lambda i:int(hist[bases[i]:bases[i]+8].sum()))]
    delta=eh-base;inside=(delta>=0)&(delta<8);delta=torch.where(inside,delta,0)
    nib=torch.tensor((0,8,16,24,4,12,20,28),device=dev).long()
    pos=torch.tensor((7,15,23,31,11,19,27,3),device=dev).long()
    code=((delta.view(r,c,32,4,8).long()<<nib)+(sign.view(r,c,32,4,8).long()<<pos)).sum(-1)
    code=mod.v18s_to_i32(code).contiguous().view(torch.uint8).view(r,c,32,16)
    lo=lo.to(torch.uint8);data=torch.stack([lo[...,:16],lo[...,16:],code],dim=2).reshape(r,c,1536).contiguous()
    er,ec,el,ev=(~inside).nonzero().unbind(1);xor=eh[er,ec,el,ev]^base
    assert torch.equal(mod.v18d_v18d_native_c12_unpack(data,base,(er,ec,el,ev,xor)),v)
    return {'data':data,'t01':base*0x01010101,'u01':0,'fmt':1,
            'esc':(er,ec,el,ev,xor,torch.zeros_like(er)),'base':base,'n_esc':er.numel()}


def stream(mod,pk,cfg,cap=64):
    # Execute a copied lab stream-builder with a larger exact exception capacity.
    # Do not change the parent module's globals, packer, or dispatch.
    namespace=dict(mod.__dict__);namespace['V18S_ESC_CAP']=cap
    exec(inspect.getsource(mod.v18s_stream),namespace)
    return namespace['v18s_stream'](pk,1,*cfg)


def audit():
    cases=0
    for base in range(0,121,8):
      for bits in range(65536):
        hi=(bits>>8)&127;sign=bits>>15;low=bits&255
        inside=base<=hi<base+8
        code=(hi-base if inside else 0)|(sign<<7)
        restored=((code&0x87)|base)<<8|low
        if not inside:restored^=(hi^base)<<8
        assert restored==bits
        cases+=1
    # Full packed-word positioning, including the rotate across bit 0.
    import random
    rng=random.Random(72);nib=(0,8,16,24,4,12,20,28);pos=(7,15,23,31,11,19,27,3)
    for base in range(0,121,8):
      for _ in range(4096):
        ds=[rng.randrange(8) for _ in range(8)];ss=[rng.randrange(2) for _ in range(8)]
        C=sum((d<<n)|(s<<p) for d,s,n,p in zip(ds,ss,nib,pos));rot=((C>>4)|(C<<28))&0xffffffff
        h=[(C&0x87878787)|(base*0x01010101),(rot&0x87878787)|(base*0x01010101)]
        assert all(((h[j//4]>>(8*(j%4)))&255)==((ss[j]<<7)|base|ds[j]) for j in range(8))
    return {'bf16_patterns_times_bases':cases,'packed_word_cases':65536,'passed':True}

if __name__=='__main__':print(audit())
