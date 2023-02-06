from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc_args = ['-maxrregcount=16']

setup(
    name='fused_adan',
    ext_modules=[
        CUDAExtension(
            'fused_adan', 
            sources=['pybind_adan.cpp','fused_adan_kernel.cu']
        #,extra_compile_args={'nvcc': nvcc_args}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })