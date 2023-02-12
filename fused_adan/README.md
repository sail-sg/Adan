# Adan Optimizer fused kernel

## Compile Requirements
1. Libtorch/Pytorch (Aten is required)
2. CUDA Toolkit

## Install
1. `cd fused_adan`
2. Using `python setup.py install` to install the fused kernel.
3. `cd ..` 
4. Using `python setup.py install` to install the Adan optimizer interface.

## Usage
Using `Adan(..., foreach=False, fused=True)` enable fused Adan kernel with single tensor access.  
Using `Adan(..., foreach=True, fused=True)` enable fused Adan kernel with multi tensor access.

`foreach=True` is recommended for better performance.

<!-- ## File Structure
```
fused_adan
├── include // fused kernel header files
├── README.md 
├── fused_adan_kernel.cu // single tensor fused kernel source files
├── multi_tensor_adan_kernel.cu // multi tensor fused kernel source files
├── pybind.cpp // pybind11 interface
└── setup.py
``` -->


## Test Results