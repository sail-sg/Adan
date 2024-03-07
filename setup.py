import os
import sys
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.cuda import is_available

build_cuda_ext = is_available() or os.getenv('FORCE_CUDA', '0') == '1'

cuda_extension = None
if "--unfused" in sys.argv:
    print("Building unfused version of adan")
    sys.argv.remove("--unfused")
elif build_cuda_ext:
    cuda_extension = CUDAExtension(
        'fused_adan', 
        sources=['fused_adan/pybind_adan.cpp', './fused_adan/fused_adan_kernel.cu', './fused_adan/multi_tensor_adan_kernel.cu']
    )

setup(
    name='adan',
    python_requires='>=3.8',
    version='0.0.2',
    install_requires=['torch'],
    py_modules=['adan'],
    description=(
        'Adan: Adaptive Nesterov Momentum Algorithm for '
        'Faster Optimizing Deep Models'
    ),
    author=(
        'Xie, Xingyu and Zhou, Pan and Li, Huan and '
        'Lin, Zhouchen and Yan, Shuicheng'
    ),
    ext_modules=[cuda_extension] if cuda_extension is not None else [],
    cmdclass={'build_ext': BuildExtension} if build_cuda_ext else {},
)