"""Lab-only pooler32 HMMA register-operand reuse experiment.

Parent source: mod.b2p_tt_v10_m32x__CUDA.
Parent state: state['bundle2']['pooler']['plans']['pooler'].
The plan contains wp, esc, t01, u01, idx, cfg, n and the parent extension.

Example (inside a lab run, after parent.prepare):
    plan = state['bundle2']['pooler']['plans']['pooler']
    ext = build(parent)
    out = torch.empty((32,4096),device=x.device,dtype=torch.bfloat16)
    launch(parent, ext, plan, x, out)

This experiment changes neither the packed format nor the number of global
weight loads. It places consecutive HMMAs using the same A operand registers
next to one another, potentially improving the hardware operand-reuse cache.
Check emitted SASS .reuse flags and timing before claiming that benefit.
"""
import hashlib


_BRANCH = '''          if constexpr (XM == 6 || XM == 8 || XM == 10) {
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
              for (int n = 0; n < MT; ++n)
#pragma unroll
                for (int j = 0; j < RT; ++j)
                  r3_pool32_pooler_d9_cap2_a_m32x_mma(acc[0][j][n], A[j] + q * 4, b[n][2*q], b[n][2*q+1]);
'''


def source(mod):
    """Return the pooler32 source with one precisely matched branch extension.

    XM7 already has b[n] in registers. For each fixed j,n accumulator, the
    sequence q=0,1,2,3 is unchanged; only independent accumulators interleave.
    Restricting the extension to !HILO and !NOMMA preserves all diagnostic and
    FP32 hi+lo paths. Both instantiated production cases are BF16 FMT0/1.
    """
    src = mod.b2p_tt_v10_m32x__CUDA
    if src.count(_BRANCH) != 1:
        raise RuntimeError('Expected exactly one complete pooler32 HMMA branch')
    replacement = _BRANCH.replace(
        'XM == 6 || XM == 8 || XM == 10',
        'XM == 6 || XM == 8 || XM == 10 || (XM == 7 && !HILO && !NOMMA)')
    return src.replace(_BRANCH, replacement, 1)


def build(mod):
    """Compile only this extension; the parent's extension/state stay intact."""
    src = source(mod)
    name = 'lab_pool32_reuse_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return mod.load_inline(
        name, mod.b2p_tt_v10_m32x__CPP, cuda_sources=src,
        functions=['r3_pool32_pooler_d9_cap2_a_m32x_launch'],
        extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'],
        no_implicit_headers=True, with_pytorch_error_handling=False)


def launch(mod, ext, plan, x, out, query=0):
    """Use the preserved packing and active CUDA stream through parent launcher.

    query=1 reports occupancy; query=2 returns numRegs*1000+localSizeBytes.
    The parent launcher obtains the stream at call time, including graph capture.
    """
    rc = mod.b2p_tt_v10_m32x_m32x_launch_raw(
        ext, plan['idx'], plan, x, out, 4096, plan['cfg'], query)
    if query:
        return rc
    if rc:
        raise RuntimeError(f'pooler32 register-reuse launch failed: {rc}')
    return out
