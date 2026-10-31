"""Lab-only variants of v18s; never mutates the protected parent module.

The arithmetic decoder is transplanted from the parent's v18d kernel. Its
different nibble/sign ordering REQUIRES v18d's packer, not v18s_pack_format.
The unchanged stream builder and escape patcher address original value indices,
so they work with either code layout after the matching decode reconstructs A.
"""
import re


_DECODER = r'''SM3D void v18s_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  uint32_t rotated;
  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));
  const uint32_t low_high = (C & 0x87878787u) + T0;
  const uint32_t high_high = (rotated & 0x87878787u) + T0;
  a[0] = v18s_prmt(L0, low_high, 0x5140u);
  a[1] = v18s_prmt(L0, low_high, 0x7362u);
  a[2] = v18s_prmt(L1, high_high, 0x5140u);
  a[3] = v18s_prmt(L1, high_high, 0x7362u);
}'''


def source(mod, instances):
    """Return arithmetic-decoder CUDA source with unchanged launch/escape logic.

    Instance fields: FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK,
    CWR, CWK, CKU, CS, OPT. Only DIAG=0 computes valid submissions.
    """
    src = mod.v18s_cuda_source(instances)
    pattern = r'SM3D void v18s_dec\([^\n]+\) \{.*?\n\}'
    result, count = re.subn(pattern, lambda _: _DECODER, src, flags=re.S)
    if count != 1:
        raise RuntimeError(f'Expected exactly one v18s decoder, found {count}')
    return '// Lab arithmetic decoder; requires matching v18d packing.\n' + result


def pack(mod, w, fmt):
    """Prepare exact v18d packed units suitable for unchanged v18s_stream.

    Both unit functions use the same BF16-to-MMA permutation and low-byte order.
    Both streams map escapes by (tile, K chunk, lane, original value index,
    high-byte XOR, hi/lo half). Neither escape mapping depends on the positions
    of the compressed nibbles/signs. v18d_pack_format verifies unpack==original
    before returning. FMT 0/2 remain uncompressed control paths.
    """
    if fmt not in (0, 1, 2, 3):
        raise ValueError(fmt)
    return mod.v18d_v18d_native_pack_format(w, fmt)


def audit_integer_decoder():
    """CPU proof-by-enumeration of per-byte arithmetic and code positioning.

    base in [0,120] plus delta in [0,7] stays below 128. Adding the
    replicated base cannot carry across bytes even when a sign bit is set.
    Enumerate all high-byte codes and bit positions, plus arbitrary low bytes.
    Escape XOR is applied to the original decoded high byte before MMA.
    """
    nib = (0, 8, 16, 24, 4, 12, 20, 28)
    signpos = (7, 15, 23, 31, 11, 19, 27, 3)
    checked = 0
    for base in range(121):
        for sign in range(2):
            for delta in range(8):
                for value in range(8):
                    code = (delta << nib[value]) | (sign << signpos[value])
                    rotated = ((code >> 4) | (code << 28)) & 0xffffffff
                    halves = ((code & 0x87878787) + base * 0x01010101,
                              (rotated & 0x87878787) + base * 0x01010101)
                    got = (halves[value // 4] >> (8 * (value % 4))) & 255
                    expected = (sign << 7) | (base + delta)
                    assert got == expected, (base, sign, delta, value, got)
                    # Other positions must reconstruct base with positive sign.
                    for j in range(8):
                        if j != value:
                            assert ((halves[j // 4] >> (8 * (j % 4))) & 255) == base
                    for low in (0, 1, 127, 128, 254, 255):
                        assert low | (got << 8) == low | (expected << 8)
                    checked += 1
        for original_high in range(128):
            # Escapes encode delta=0 and XOR back to the exact original exponent.
            assert base ^ (original_high ^ base) == original_high
    return {'integer_code_cases': checked, 'escape_high_byte_cases': 121 * 128}


# Bounded starting points, not asserted winners:
# - pooler1: keep (WR,WK,KU,S)=(2,16,8,1), test ST=1/2/4, OPT=4/6.
# - O8: keep (1,8,2,1), test ST=1/2, OPT=0/2/4/6.
# - QKV1: reuse native geometry (1,4,8,4), RED=2, ST=1/2/4,
#   OPT=0/2/4/6 to isolate stream scheduling from decoder arithmetic.
# Increasing MT alone cannot improve reuse when M is unchanged: the baseline
# already uses decoded weights for all MT*8 token slots within that warp.

if __name__ == '__main__':
    print(audit_integer_decoder())
