from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import platform
import ctypes.util

def has_cuda():
    # Check if the system has a CUDA-capable GPU
    has_gpu = any(['nvidia' in x.lower() for x in platform.uname()._field_defaults['node']])

    # Check if CUDA library is installed
    cuda_library = ctypes.util.find_library('cuda')
    has_cuda_library = cuda_library is not None

    return has_gpu and has_cuda_library

cuda_extension = CUDAExtension(
            'fused_adan', 
            sources=['fused_adan/pybind_adan.cpp','./fused_adan/fused_adan_kernel.cu', './fused_adan/multi_tensor_adan_kernel.cu']
        )

setup(
    name='adan',
    python_requires='>=3.8',
    version='0.0.1',
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
    ext_modules=[cuda_extension] if has_cuda() else [],
    cmdclass={'build_ext': BuildExtension} if has_cuda() else {},
)
