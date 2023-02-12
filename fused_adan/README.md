# Adan Optimizer fused kernel

## Dependence
1. Libtorch/Pytorch (ATen is required, Compile passed on Pytorch 1.13.1)
2. CUDA Toolkit (Compile passed on CUDA 11.6)
3. ninja

## Usage
Using `Adan(..., foreach=False, fused=True)` enable fused Adan kernel with single tensor access.  
Using `Adan(..., foreach=True, fused=True)` enable fused Adan kernel with multi tensor access.

`foreach=True` is recommended for better performance.

## Test Results