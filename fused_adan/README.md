# Adan Optimizer fused kernel

## Dependence
1. Libtorch/Pytorch (ATen is required, Compile passed on Pytorch 1.13.1)
2. CUDA Toolkit (Compile passed on CUDA 11.6)
3. ninja

## Usage
Using `Adan(..., foreach=False, fused=True)` enable fused Adan kernel with single tensor access.  
Using `Adan(..., foreach=True, fused=True)` enable fused Adan kernel with multi tensor access.

`foreach=True` is recommended for better performance.

**Single tensor access**
A *for loop* is used to traverse each layer when calculating the gradient of each Layer, requiring multiple kernel starts.
In theory, accessing only one layer of parameters at a time is good for reducing peak memory usage, but it introduces a kernel startup overhead.

**Multi tensor access**
The parameters of all layers are passed into the kernel at once, and the kernel internally uses a for loop to traverse each layer, requiring only one kernel start.
Theoretically, this will lead to an increase in peak memory usage, but will reduce the overhead of kernel startup.
In actual tests, the increase in memory usage is not significant, but the kernel startup overhead is much reduced.

## Test Results