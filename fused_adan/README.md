# Adan Optimizer fused kernel

## Compile Requirements
1. Libtorch
2. CUDA Toolkit

## Install
1. `cd fused_adan`
2. Using `python setup.py install` to install the fused kernel.
3. `cd ..` 
4. Using `python setup.py install` to install the Adan optimizer interface.

## Usage
Using `Adan(..., foreach=False, fused=True)` enable fused Adan kernel with single tensor access.  
Using `Adan(..., foreach=True, fused=True)` enable fused Adan kernel with multi tensor access.

`foreach=True` is recommended for better performance, with higher memory demand.

## Test Results on Tesla T4, Mnist Dataset, simple 2 Layers CNN network
Achieved active warps occupancy:
* for single tensor access unfused: 36.53 %
* for multi tensor access unfused: 41.96 %
* for single tensor access fused: 43.09 %
* for multi tensor access fused: 44.84 %

Wall Duration for *adan.py: step()*:
* for single tensor access unfused: Avg. 2.82ms
* for multi tensor access unfused: Avg. 0.86ms
* for single tensor access fused: Avg. 1.24ms
* for multi tensor access fused: Avg. 0.33ms