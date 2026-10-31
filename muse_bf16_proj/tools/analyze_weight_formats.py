"""CPU analysis of the public weight generator; no GPU, imports, or kernel changes.

CPU and CUDA RNG streams differ. These are full-size samples of the specified
distribution, not a claim to reproduce held-out judge tensors. GPU testing must
recheck packed sizes on its own tensors.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import reference


def window(hist, width):
    counts = np.convolve(hist, np.ones(width, dtype=np.int64), 'valid')
    base = int(counts.argmax())
    return base, int(counts[base])


def analyze(bits):
    bits = bits.reshape(-1).astype(np.uint16)
    count = bits.size
    exp = (bits >> 7) & 255
    hist = np.bincount(exp, minlength=256)
    base, covered = window(hist, 7)
    exceptions = count - covered
    # Concrete ZipServ-like accounting: 64-value fragments, each common and
    # exception payload separately aligned to 16B; 2 uint32 offsets per fragment.
    common = ((exp >= base) & (exp < base + 7)).reshape(-1, 64).sum(1)
    padded = 24 + ((common + 15)//16)*16 + ((2*(64-common)+15)//16)*16 + 8
    # Amortized alternative: 1024-value groups, 16B alignment, two offsets.
    common_big = common.reshape(-1,16).sum(1)
    grouped = 384 + ((common_big+15)//16)*16 + ((2*(1024-common_big)+15)//16)*16 + 8
    high = (bits >> 8) & 127
    high_hist = np.bincount(high, minlength=128)
    b4, c4 = window(high_hist,4)
    b8, c8 = window(high_hist,8)
    # Keep all low bytes and sign, encode high-byte delta in two bits.
    # Sparse 32-bit XOR patches ordered by warp/iteration; one uint32 offset
    # per warp (8192 weights for the protected pooler schedule).
    regular11 = count*11//8 + (count-c4)*4 + (count//8192+1)*4
    adaptive = {}
    rng = np.random.default_rng(93451)
    for label, ex in [('original',exp), ('shuffled',rng.permutation(exp))]:
        # Global rank code, tile-local width, zero sentinel, full BF16 escapes.
        ranks = np.empty(256,dtype=np.int16)
        ranks[np.argsort(-hist,kind='stable')] = np.arange(256)
        rank = ranks[ex]
        for size in (128,256):
            rr=rank.reshape(-1,size)
            costs=[]
            for width in (2,3,4):
                ne=(rr >= (1<<width)-1).sum(1)
                # 1-byte width + 4-byte byte offset, 16-byte payload alignment.
                payload=size + (size*width)//8 + 2*ne
                costs.append(((payload+15)//16)*16+5)
            adaptive[f'{label}_{size}'] = float(np.min(costs,axis=0).sum()*8/count)
    p=hist[hist>0]/count
    return {'weights':count,'exponent_histogram':hist.tolist(),
            'exponent_entropy_bits':float(-(p*np.log2(p)).sum()),
            'seven_exponent_base':base,'seven_exponent_coverage':covered/count,
            'zip_payload_bpw':11+8*exceptions/count,
            'zip_64_aligned_with_offsets_bpw':float(padded.sum()*8/count),
            'zip_1024_aligned_with_offsets_bpw':float(grouped.sum()*8/count),
            'regular11_high_base':b4,'regular11_exception_rate':1-c4/count,
            'regular11_total_bpw':regular11*8/count,
            'baseline12_high_base':b8,'baseline12_exception_rate':1-c8/count,
            'baseline12_pooler_schedule_bpw':12+128*8/8192,
            'adaptive_rank_layout_bpw':adaptive}


def main():
    torch.set_num_threads(4)
    result={'description':__doc__,'torch':str(torch.__version__),'samples':[]}
    for seed in (5151,82731,294817):
        weights=reference.generate_weights(seed,device='cpu')
        for layer,w in weights.items():
            if layer=='router': continue
            row={'seed':seed,'layer':layer,**analyze(w.view(torch.int16).numpy())}
            result['samples'].append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='exponent_histogram'}),flush=True)
        del weights
    dest=ROOT/'results/local/weight-format-analysis.json'
    dest.parent.mkdir(exist_ok=True,parents=True)
    dest.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
