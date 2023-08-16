/* Copyright NVIDIA/apex
   Copyright AlexwellChen
   This kernel is adapted from NVIDIA/apex.
*/
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

#include "include/type_shim.h" // Used for DISPATCH
#include "include/multi_tensor_apply.cuh" 
#include "include/fused_adan_kernel.cuh"

#define BLOCK_SIZE 512
#define ILP 4

using MATH_T = float;

template<typename T>
struct AdanFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<6>& tl,
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
    const float clip_global_grad_norm
    )
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];

    // potentially use to pass in list of scalar
    // int tensor_num = tl.start_tensor_this_launch + tensor_loc;

    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T* p = (T*)tl.addresses[0][tensor_loc];
    p += chunk_idx*chunk_size;

    T* g = (T*)tl.addresses[1][tensor_loc];
    g += chunk_idx*chunk_size;

    T* exp_avg = (T*)tl.addresses[2][tensor_loc];
    exp_avg += chunk_idx*chunk_size;

    T* exp_avg_sq = (T*)tl.addresses[3][tensor_loc];
    exp_avg_sq += chunk_idx*chunk_size;

    T* exp_avg_diff = (T*)tl.addresses[4][tensor_loc];
    exp_avg_diff += chunk_idx*chunk_size;

    T* neg_grad = (T*)tl.addresses[5][tensor_loc];
    neg_grad += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_p[ILP];
      MATH_T r_g[ILP];
      MATH_T r_exp_avg[ILP];
      MATH_T r_exp_avg_sq[ILP];
      MATH_T r_exp_avg_diff[ILP];
      MATH_T r_neg_grad_diff[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_p[ii] = p[i];
          r_g[ii] = g[i];
          r_exp_avg[ii] = exp_avg[i];
          r_exp_avg_sq[ii] = exp_avg_sq[i];
          r_exp_avg_diff[ii] = exp_avg_diff[i];
          r_neg_grad_diff[ii] = neg_grad[i];
        } else {
          r_p[ii] = MATH_T(0);
          r_g[ii] = MATH_T(0);
          r_exp_avg[ii] = MATH_T(0);
          r_exp_avg_sq[ii] = MATH_T(0);
          r_exp_avg_diff[ii] = MATH_T(0);
          r_neg_grad_diff[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        r_g[ii] *= clip_global_grad_norm; //scaled_grad
        MATH_T update;
        r_neg_grad_diff[ii] = r_g[ii] + r_neg_grad_diff[ii];
        update = r_g[ii] + beta2 * r_neg_grad_diff[ii]; // 1 MAC, reused twice

        r_exp_avg[ii] = beta1 * r_exp_avg[ii] + (1 - beta1) * r_g[ii];
        r_exp_avg_diff[ii] = beta2 * r_exp_avg_diff[ii] + (1 - beta2) * r_neg_grad_diff[ii];
        
        r_exp_avg_sq[ii] = beta3 * r_exp_avg_sq[ii] + (1 - beta3) * update * update;

        MATH_T denom;
        denom = sqrtf(r_exp_avg_sq[ii]) / bias_correction3_sqrt + epsilon;
        MATH_T step_size_diff = lr * beta2 / bias_correction2;
        MATH_T step_size = lr / bias_correction1;

        if(no_prox){
          r_p[ii] = r_p[ii] * (1 - lr * decay);
          r_p[ii] = r_p[ii] - step_size * r_exp_avg[ii] / denom;
          r_p[ii] = r_p[ii] - step_size_diff * r_exp_avg_diff[ii] / denom;
        } else {
          r_p[ii] = r_p[ii] - step_size * r_exp_avg[ii] / denom;
          r_p[ii] = r_p[ii] - step_size_diff * r_exp_avg_diff[ii] / denom;
          r_p[ii] = r_p[ii] / (1 + lr * decay);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          g[i] = r_g[ii];
          p[i] = r_p[ii];
          exp_avg[i] = r_exp_avg[ii];
          exp_avg_sq[i] = r_exp_avg_sq[ii];
          exp_avg_diff[i] = r_exp_avg_diff[ii];
        }
      }
    }
  }
};

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
  const float clip_global_grad_norm)
{
  using namespace at;
  TORCH_CHECK(!tensor_lists.empty(), "tensor list cannot be empty")
  if (tensor_lists[0].empty()) {
    return;
  }

  // Assume single type across p,g,m1,m2 now
  DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(
    tensor_lists[0][0].scalar_type(), 0, "adan",
    multi_tensor_apply<6>(
      BLOCK_SIZE,
      chunk_size,
      noop_flag,
      tensor_lists,
      AdanFunctor<scalar_t_0>(),
      beta1,
      beta2,
      beta3,
      bias_correction1,
      bias_correction2,
      bias_correction3_sqrt,
      lr,
      decay,
      epsilon,
      no_prox,
      clip_global_grad_norm
      ); )

  AT_CUDA_CHECK(cudaGetLastError());

}
