/* Copyright 2021 The LightSeq Team
   Copyright NVIDIA/apex
   This apex_adam_cuda_kernel is adapted from NVIDIA/apex
*/
#include <ATen/ATen.h>
#include <torch/extension.h>

#define DISPATCH_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...)               \
  switch (TYPE) {                                                     \
    case at::ScalarType::Float: {                                     \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Half: {                                      \
      using scalar_t_##LEVEL = at::Half;                              \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }

#define DISPATCH_DOUBLE_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...)        \
  switch (TYPE) {                                                     \
    case at::ScalarType::Double: {                                    \
      using scalar_t_##LEVEL = double;                                \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Float: {                                     \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Half: {                                      \
      using scalar_t_##LEVEL = at::Half;                              \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }

#define DISPATCH_DOUBLE_AND_FLOAT(TYPE, LEVEL, NAME, ...)             \
  switch (TYPE) {                                                     \
    case at::ScalarType::Double: {                                    \
      using scalar_t_##LEVEL = double;                                \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case at::ScalarType::Float: {                                     \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
  }

// CUDA forward declaration
void fused_adan_cuda(at::Tensor& p, at::Tensor& p_copy, at::Tensor& g, at::Tensor& exp_avg, 
          at::Tensor& exp_avg_sq, at::Tensor& exp_avg_diff,
          at::Tensor& neg_grad, float beta1, float beta2, float beta3, 
          float bias_correction1, float bias_correction2, float bias_correction3_sqrt, 
          float lr, float decay, float eps, bool no_prox, float clip_global_grad_norm);