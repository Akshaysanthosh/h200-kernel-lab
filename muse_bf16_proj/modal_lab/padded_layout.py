"""Lab-only cuBLASLt weight leading-dimension support; exact values, same GEMM."""
def source(mod):
 s=mod.tt_lt__LT_CPP
 # Preserve dimensions/transposes; change ONLY physical weight leading dimension.
 old='is_packed ? n : k));'
 assert s.count(old)==2
 s=s.replace(old,'weight.stride(0)));')
 # Avoid pybind C++ type collisions with the protected parent's extension.
 s=s.replace('LtContext','PaddedContext').replace('LtPlan','PaddedPlan')
 return s
