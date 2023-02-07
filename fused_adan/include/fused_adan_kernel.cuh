/* Copyright 2021 The LightSeq Team
   Copyright NVIDIA/apex
   Copyright AlexwellChen
   This kernel is adapted from NVIDIA/apex and LightSeq Team
*/
#include <ATen/ATen.h>
#include <torch/extension.h>

// CUDA forward declaration
void fused_adan_cuda(
    at::Tensor& p, at::Tensor& p_copy, at::Tensor& g, at::Tensor& exp_avg, 
    at::Tensor& exp_avg_sq, at::Tensor& exp_avg_diff,
    at::Tensor& neg_grad, float beta1, float beta2, float beta3, 
    float bias_correction1, float bias_correction2, float bias_correction3_sqrt, 
    float lr, float decay, float eps, bool no_prox, float clip_global_grad_norm);

void multi_tensor_adan_cuda(
    int chunk_size,
    at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists,
    const float beta1,
    const float beta2,
    const float beta3,
    const float bias_correction1,
    const float bias_correction2,
    const float bias_correction3_sqrt,
    const float lr,
    const float decay,
    const float epsilon,
    const bool no_prox,
    const float clip_global_grad_norm);