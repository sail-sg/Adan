/* Copyright 2021 The LightSeq Team
   Copyright NVIDIA/apex
   This apex_adam_cuda_kernel is adapted from NVIDIA/apex
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/Exceptions.h>
#include "fused_adan_kernel.cuh"


template <typename T, typename GRAD_T>
__global__ void adan_cuda_kernel(
    T* __restrict__ p,
    GRAD_T* __restrict__ p_copy,  // For mixed precision training, pass NULL if
                                  // not needed
    const GRAD_T* __restrict__ g, T* __restrict__ exp_avg, T* __restrict__ exp_avg_sq, T* __restrict__ exp_avg_diff,
    const GRAD_T* __restrict__ neg_grad, const float b1, const float b2, const float b3, 
    const float bias_correction1, const float bias_correction2, const float bias_correction3_sqrt,
    const float lr, const float decay, const float eps, const bool no_prox, const float clip_global_grad_norm, const size_t total_size
    ){
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id >= total_size) return;

    T scaled_grad = g[global_id] * clip_global_grad_norm;

    GRAD_T diff, update;

    diff = scaled_grad + neg_grad[global_id];
    update = scaled_grad + b2 * diff;

    exp_avg[global_id] = b1 * exp_avg[global_id] + (1 - b1) * scaled_grad;

    exp_avg_diff[global_id] = b2 * exp_avg_diff[global_id] + (1 - b2) * diff;

    exp_avg_sq[global_id] = b3 * exp_avg_sq[global_id] + (1 - b3) * update * update;

    float denom;
    denom = sqrtf(exp_avg_sq[global_id]) / bias_correction3_sqrt + eps;
    update = (exp_avg[global_id] / bias_correction1 + b2 * exp_avg_diff[global_id] / bias_correction2) / denom;
    
    if (no_prox){
        p[global_id] = p[global_id] * (1 - lr * decay) + update * (-lr);
    }else{
        p[global_id] = p[global_id] + update * (-lr) / (1 + lr * decay);
    } 
    if (p_copy != NULL) p_copy[global_id] = (GRAD_T)p[global_id];
}

template <>
__global__ void adan_cuda_kernel<float, float>(
    float* __restrict__ p,
    float* __restrict__ p_copy,  // For mixed precision training, pass NULL if
                                  // not needed
    const float* __restrict__ g, float* __restrict__ exp_avg, float* __restrict__ exp_avg_sq, float* __restrict__ exp_avg_diff,
    const float* __restrict__ neg_grad, const float b1, const float b2, const float b3, 
    const float bias_correction1, const float bias_correction2, const float bias_correction3_sqrt,
    const float lr, const float decay, const float eps, const bool no_prox, const float clip_global_grad_norm, const size_t total_size){

        int global_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (global_id * 4 >= total_size) return;

        float4* p4_ptr = reinterpret_cast<float4*>(p);
        const float4* g4_ptr = reinterpret_cast<const float4*>(g);
        const float4* neg_grad4_ptr = reinterpret_cast<const float4*>(neg_grad);
        float4* exp_avg4_ptr = reinterpret_cast<float4*>(exp_avg);
        float4* exp_avg_sq4_ptr = reinterpret_cast<float4*>(exp_avg_sq);
        float4* exp_avg_diff4_ptr = reinterpret_cast<float4*>(exp_avg_diff);
        
        float4 p4 = p4_ptr[global_id];
        const float4 g4 = g4_ptr[global_id];
        const float4 neg_grad4 = neg_grad4_ptr[global_id];
        float4 exp_avg4 = exp_avg4_ptr[global_id];
        float4 exp_avg_sq4 = exp_avg_sq4_ptr[global_id];
        float4 exp_avg_diff4 = exp_avg_diff4_ptr[global_id];

        float4 new_p4;
        float4 new_exp_avg4;
        float4 new_exp_avg_sq4;
        float4 new_exp_avg_diff4;

        // float scaled_grad1 = g4.x * clip_global_grad_norm;
        // float scaled_grad2 = g4.y * clip_global_grad_norm;
        // float scaled_grad3 = g4.z * clip_global_grad_norm;
        // float scaled_grad4 = g4.w * clip_global_grad_norm;

        float scaled_grad1 = __fmul_rn(g4.x, clip_global_grad_norm);
        float scaled_grad2 = __fmul_rn(g4.y, clip_global_grad_norm);
        float scaled_grad3 = __fmul_rn(g4.z, clip_global_grad_norm);
        float scaled_grad4 = __fmul_rn(g4.w, clip_global_grad_norm);

        // float diff1 = scaled_grad1 + neg_grad4.x;
        // float diff2 = scaled_grad2 + neg_grad4.y;
        // float diff3 = scaled_grad3 + neg_grad4.z;
        // float diff4 = scaled_grad4 + neg_grad4.w;

        float diff1 = __fadd_rn(scaled_grad1, neg_grad4.x);
        float diff2 = __fadd_rn(scaled_grad2, neg_grad4.y);
        float diff3 = __fadd_rn(scaled_grad3, neg_grad4.z);
        float diff4 = __fadd_rn(scaled_grad4, neg_grad4.w);

        // float update1 = scaled_grad1 + b2 * diff1;
        // float update2 = scaled_grad2 + b2 * diff2;
        // float update3 = scaled_grad3 + b2 * diff3;
        // float update4 = scaled_grad4 + b2 * diff4;

        float update1 = __fmaf_rn(b2, diff1, scaled_grad1);
        float update2 = __fmaf_rn(b2, diff2, scaled_grad2);
        float update3 = __fmaf_rn(b2, diff3, scaled_grad3);
        float update4 = __fmaf_rn(b2, diff4, scaled_grad4);

        // new_exp_avg4.x = b1 * exp_avg4.x + (1 - b1) * scaled_grad1;
        // new_exp_avg4.y = b1 * exp_avg4.y + (1 - b1) * scaled_grad2;
        // new_exp_avg4.z = b1 * exp_avg4.z + (1 - b1) * scaled_grad3;
        // new_exp_avg4.w = b1 * exp_avg4.w + (1 - b1) * scaled_grad4;

        float one_minus_b1 = 1 - b1;
        float one_minus_b2 = 1 - b2;
        float one_minus_b3 = 1 - b3;

        new_exp_avg4.x = __fmaf_rn(b1, scaled_grad1, __fmul_rn(exp_avg4.x, one_minus_b1));
        new_exp_avg4.y = __fmaf_rn(b1, scaled_grad2, __fmul_rn(exp_avg4.y, one_minus_b1));
        new_exp_avg4.z = __fmaf_rn(b1, scaled_grad3, __fmul_rn(exp_avg4.z, one_minus_b1));
        new_exp_avg4.w = __fmaf_rn(b1, scaled_grad4, __fmul_rn(exp_avg4.w, one_minus_b1));


        // new_exp_avg_sq4.x = b3 * exp_avg_sq4.x + (1 - b3) * update1 * update1;
        // new_exp_avg_sq4.y = b3 * exp_avg_sq4.y + (1 - b3) * update2 * update2;
        // new_exp_avg_sq4.z = b3 * exp_avg_sq4.z + (1 - b3) * update3 * update3;
        // new_exp_avg_sq4.w = b3 * exp_avg_sq4.w + (1 - b3) * update4 * update4;

        new_exp_avg_sq4.x = __fmaf_rn(b3, __powf(update1, 2), __fmul_rn(exp_avg_sq4.x, one_minus_b3));
        new_exp_avg_sq4.y = __fmaf_rn(b3, __powf(update2, 2), __fmul_rn(exp_avg_sq4.y, one_minus_b3));
        new_exp_avg_sq4.z = __fmaf_rn(b3, __powf(update3, 2), __fmul_rn(exp_avg_sq4.z, one_minus_b3));
        new_exp_avg_sq4.w = __fmaf_rn(b3, __powf(update4, 2), __fmul_rn(exp_avg_sq4.w, one_minus_b3));

        // new_exp_avg_diff4.x = b2 * exp_avg_diff4.x + (1 - b2) * diff1;
        // new_exp_avg_diff4.y = b2 * exp_avg_diff4.y + (1 - b2) * diff2;
        // new_exp_avg_diff4.z = b2 * exp_avg_diff4.z + (1 - b2) * diff3;
        // new_exp_avg_diff4.w = b2 * exp_avg_diff4.w + (1 - b2) * diff4;

        new_exp_avg_diff4.x = __fmaf_rn(b2, diff1, __fmul_rn(exp_avg_diff4.x, one_minus_b2));
        new_exp_avg_diff4.y = __fmaf_rn(b2, diff2, __fmul_rn(exp_avg_diff4.y, one_minus_b2));
        new_exp_avg_diff4.z = __fmaf_rn(b2, diff3, __fmul_rn(exp_avg_diff4.z, one_minus_b2));
        new_exp_avg_diff4.w = __fmaf_rn(b2, diff4, __fmul_rn(exp_avg_diff4.w, one_minus_b2));

        float4 denom4;

        // denom4.x = sqrt(new_exp_avg_sq4.x - new_exp_avg_diff4.x * new_exp_avg_diff4.x / b2) + eps;
        // denom4.y = sqrt(new_exp_avg_sq4.y - new_exp_avg_diff4.y * new_exp_avg_diff4.y / b2) + eps;
        // denom4.z = sqrt(new_exp_avg_sq4.z - new_exp_avg_diff4.z * new_exp_avg_diff4.z / b2) + eps;
        // denom4.w = sqrt(new_exp_avg_sq4.w - new_exp_avg_diff4.w * new_exp_avg_diff4.w / b2) + eps;

        // use __fadd_rn, __fsqrt_rn, __fdiv_rn, __fmul_rn, __fmaf_rn
        denom4.x = __fadd_rn(__fsqrt_rn(__fsub_rn(new_exp_avg_sq4.x, __fdiv_rn(__fmul_rn(new_exp_avg_diff4.x, new_exp_avg_diff4.x), b2))), eps);
        denom4.y = __fadd_rn(__fsqrt_rn(__fsub_rn(new_exp_avg_sq4.y, __fdiv_rn(__fmul_rn(new_exp_avg_diff4.y, new_exp_avg_diff4.y), b2))), eps);
        denom4.z = __fadd_rn(__fsqrt_rn(__fsub_rn(new_exp_avg_sq4.z, __fdiv_rn(__fmul_rn(new_exp_avg_diff4.z, new_exp_avg_diff4.z), b2))), eps);
        denom4.w = __fadd_rn(__fsqrt_rn(__fsub_rn(new_exp_avg_sq4.w, __fdiv_rn(__fmul_rn(new_exp_avg_diff4.w, new_exp_avg_diff4.w), b2))), eps);

        // update1 = (new_exp_avg4.x / bias_correction1 + b2 * new_exp_avg_diff4.x / bias_correction2) / denom4.x;
        // update2 = (new_exp_avg4.y / bias_correction1 + b2 * new_exp_avg_diff4.y / bias_correction2) / denom4.y;
        // update3 = (new_exp_avg4.z / bias_correction1 + b2 * new_exp_avg_diff4.z / bias_correction2) / denom4.z;
        // update4 = (new_exp_avg4.w / bias_correction1 + b2 * new_exp_avg_diff4.w / bias_correction2) / denom4.w;

        // use __fadd_rn, __fsqrt_rn, __fdiv_rn, __fmul_rn, __fmaf_rn
        update1 = __fdiv_rn(__fadd_rn(__fdiv_rn(new_exp_avg4.x, bias_correction1), __fdiv_rn(__fmul_rn(b2, new_exp_avg_diff4.x), bias_correction2)), denom4.x);
        update2 = __fdiv_rn(__fadd_rn(__fdiv_rn(new_exp_avg4.y, bias_correction1), __fdiv_rn(__fmul_rn(b2, new_exp_avg_diff4.y), bias_correction2)), denom4.y);
        update3 = __fdiv_rn(__fadd_rn(__fdiv_rn(new_exp_avg4.z, bias_correction1), __fdiv_rn(__fmul_rn(b2, new_exp_avg_diff4.z), bias_correction2)), denom4.z);
        update4 = __fdiv_rn(__fadd_rn(__fdiv_rn(new_exp_avg4.w, bias_correction1), __fdiv_rn(__fmul_rn(b2, new_exp_avg_diff4.w), bias_correction2)), denom4.w);

        if (no_prox){
            // new_p4.x = p4.x * (1 - lr * decay) + update1 * (-lr);
            // new_p4.y = p4.y * (1 - lr * decay) + update2 * (-lr);
            // new_p4.z = p4.z * (1 - lr * decay) + update3 * (-lr);
            // new_p4.w = p4.w * (1 - lr * decay) + update4 * (-lr);
            // use __fadd_rn, __fsqrt_rn, __fdiv_rn, __fmul_rn, __fmaf_rn
            new_p4.x = __fadd_rn(__fmul_rn(p4.x, __fsub_rn(1, __fmul_rn(lr, decay))), __fmul_rn(update1, __fsub_rn(0, lr)));
            new_p4.y = __fadd_rn(__fmul_rn(p4.y, __fsub_rn(1, __fmul_rn(lr, decay))), __fmul_rn(update2, __fsub_rn(0, lr)));
            new_p4.z = __fadd_rn(__fmul_rn(p4.z, __fsub_rn(1, __fmul_rn(lr, decay))), __fmul_rn(update3, __fsub_rn(0, lr)));
            new_p4.w = __fadd_rn(__fmul_rn(p4.w, __fsub_rn(1, __fmul_rn(lr, decay))), __fmul_rn(update4, __fsub_rn(0, lr)));
        }else{
            // new_p4.x = p4.x + update1 * (-lr) / (1 + lr * decay);
            // new_p4.y = p4.y + update2 * (-lr) / (1 + lr * decay);
            // new_p4.z = p4.z + update3 * (-lr) / (1 + lr * decay);
            // new_p4.w = p4.w + update4 * (-lr) / (1 + lr * decay);
            // use __fadd_rn, __fsqrt_rn, __fdiv_rn, __fmul_rn, __fmaf_rn
            new_p4.x = __fadd_rn(p4.x, __fdiv_rn(__fmul_rn(update1, __fsub_rn(0, lr)), __fadd_rn(1, __fmul_rn(lr, decay))));
            new_p4.y = __fadd_rn(p4.y, __fdiv_rn(__fmul_rn(update2, __fsub_rn(0, lr)), __fadd_rn(1, __fmul_rn(lr, decay))));
            new_p4.z = __fadd_rn(p4.z, __fdiv_rn(__fmul_rn(update3, __fsub_rn(0, lr)), __fadd_rn(1, __fmul_rn(lr, decay))));
            new_p4.w = __fadd_rn(p4.w, __fdiv_rn(__fmul_rn(update4, __fsub_rn(0, lr)), __fadd_rn(1, __fmul_rn(lr, decay))));
        }   

        p4_ptr[global_id] = new_p4;
        exp_avg4_ptr[global_id] = new_exp_avg4;
        exp_avg_sq4_ptr[global_id] = new_exp_avg_sq4;
        exp_avg_diff4_ptr[global_id] = new_exp_avg_diff4;
}

void fused_adan_cuda(at::Tensor& p, at::Tensor& p_copy, at::Tensor& g, at::Tensor& exp_avg, 
          at::Tensor& exp_avg_sq, at::Tensor& exp_avg_diff,
          at::Tensor& neg_grad, float beta1, float beta2, float beta3, 
          float bias_correction1, float bias_correction2, float bias_correction3_sqrt, 
          float lr, float decay, float eps, bool no_prox, float clip_global_grad_norm){
    // Get tensor size
    int total_size = p.numel();
    AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p),
              "parameter tensor is too large to be indexed with int32");
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (g.scalar_type() == at::ScalarType::Half) {
        const int block_dim = 1024;
        int grid_dim = ((total_size + block_dim - 1) / block_dim);
        const dim3 blocks(grid_dim);
        // all other values should be fp32 for half gradients
        AT_ASSERTM(p.scalar_type() == at::ScalarType::Float,
                  "expected parameter to be of float type");
        // dispatch is done on the gradient type
        using namespace at;  // prevents "toString is undefined" errors
        DISPATCH_FLOAT_AND_HALF(
            g.scalar_type(), 0, "adan_cuda_kernel",
            using accscalar_t = at::acc_type<scalar_t_0, true>;
            adan_cuda_kernel<accscalar_t, scalar_t_0>
            <<<blocks, block_dim, 0, stream>>>(
                p.data_ptr<accscalar_t>(),
                p_copy.numel() ? p_copy.data_ptr<scalar_t_0>() : NULL,
                g.data_ptr<scalar_t_0>(), exp_avg.data_ptr<accscalar_t>(), exp_avg_sq.data_ptr<accscalar_t>(),exp_avg_diff.data_ptr<accscalar_t>(), 
                neg_grad.data_ptr<scalar_t_0>(), 
                beta1, beta2, beta3, bias_correction1, bias_correction2, bias_correction3_sqrt, 
                lr, decay, eps, no_prox, clip_global_grad_norm, total_size
                );
            );
    } else {
        using namespace at;
        const int block_dim = 1024;
        int grid_dim = ((total_size + block_dim - 1) / block_dim) >> 2;
        if (grid_dim == 0) grid_dim = 1;
        const dim3 blocks(grid_dim);
        DISPATCH_DOUBLE_AND_FLOAT(
            g.scalar_type(), 0, "adan_cuda_kernel",
            adan_cuda_kernel<scalar_t_0, scalar_t_0>
            <<<blocks, block_dim, 0, stream>>>(
                p.data_ptr<scalar_t_0>(),
                NULL,
                g.data_ptr<scalar_t_0>(), exp_avg.data_ptr<scalar_t_0>(), exp_avg_sq.data_ptr<scalar_t_0>(),exp_avg_diff.data_ptr<scalar_t_0>(), 
                neg_grad.data_ptr<scalar_t_0>(), 
                beta1, beta2, beta3, bias_correction1, bias_correction2, bias_correction3_sqrt, 
                lr, decay, eps, no_prox, clip_global_grad_norm, total_size
            );
        );
    }
    AT_CUDA_CHECK(cudaGetLastError());
}

