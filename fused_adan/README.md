# Adan Optimizer fused kernel

## Install
1. Using `python fused_adan/setup.py install` to install the fused kernel
2. Using `python setup.py install` to install the Adan optimizer.

## Usage
Using `Adan(..., foreach=False, fused=True)` enable fused Adan kernel.

## Limitation
Currently using single tensor, each layer will call adan kernel once.
Multi tensor version is under development.