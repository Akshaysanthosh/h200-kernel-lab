"""Compile the host wrapper to an object; does NOT link/run CUDA or validate GPU math."""
import json
from pathlib import Path
import site
import subprocess
import sysconfig

import torch
from torch.utils.cpp_extension import include_paths

root = Path('/work')
includes = include_paths() + [sysconfig.get_paths()['include']]
for package in ('cuda_runtime', 'cuda_nvcc', 'cuda_cccl', 'cublas'):
    matches = [Path(p) / 'nvidia' / package / 'include' for p in site.getsitepackages()]
    includes.append(str(next(p for p in matches if p.is_dir())))
command = ['g++', '-std=c++17', '-fPIC', '-O0', '-Werror=return-type',
           '-DTORCH_EXTENSION_NAME=okc_lt_compile_probe',
           # CPU wheels omit CUDA's generated CMake macro file. This supported
           # header switch is for this Linux compile-only probe, not submission.
           '-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE',
           f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}']
command += [f'-I{p}' for p in includes]
command += ['-c', str(root / 'cuBLASLt/lt_plan.cpp'), '-o', '/tmp/okc_lt_plan.o']
result = subprocess.run(command)
if result.returncode:
    raise SystemExit(result.returncode)
print(json.dumps({'object_compile_passed': True, 'torch_headers': torch.__version__,
                  'cuda_headers': '12.8', 'gpu_execution': False,
                  'cpu_wheel_cuda_macro_file_bypassed': True,
                  'link_validation': False, 'object_bytes': Path('/tmp/okc_lt_plan.o').stat().st_size}))
