#include <torch/extension.h>

#include "../include/fused_adan_kernel.cuh"

// x is torch::Tensor
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// C++ interface

void adan(at::Tensor& p, 
          at::Tensor& p_copy, 
          at::Tensor& g, 
          at::Tensor& exp_avg, 
          at::Tensor& exp_avg_sq, 
          at::Tensor& exp_avg_diff,
          at::Tensor& pre_g, 
          float beta1, float beta2, float beta3, 
          float bias_correction1, float bias_correction2, float bias_correction3_sqrt, 
          float lr, float decay, float eps, bool no_prox, float grad_scale) {
  CHECK_INPUT(p);
  if (p_copy.numel() > 0) CHECK_INPUT(p_copy);
  CHECK_INPUT(exp_avg);
  CHECK_INPUT(exp_avg_sq);
  CHECK_INPUT(exp_avg_diff);
  CHECK_INPUT(g);
  CHECK_INPUT(pre_g);
  int64_t num_elem = p.numel();
  AT_ASSERTM(exp_avg.numel() == num_elem,
             "number of elements in exp_avg and p tensors should be equal");
  AT_ASSERTM(exp_avg_sq.numel() == num_elem,
             "number of elements in exp_avg_sq and p tensors should be equal");
  AT_ASSERTM(exp_avg_diff.numel() == num_elem,
             "number of elements in exp_avg_diff and p tensors should be equal");
  AT_ASSERTM(g.numel() == num_elem,
             "number of elements in g and p tensors should be equal");
  AT_ASSERTM(pre_g.numel() == num_elem,
             "number of elements in pre_g and p tensors should be equal");
  AT_ASSERTM(p_copy.numel() == num_elem || p_copy.numel() == 0,
             "number of elements in p_copy and p tensors should be equal, or "
             "p_copy should be empty");

  fused_adan_cuda(p, p_copy, g, 
                  exp_avg, exp_avg_sq, exp_avg_diff,
                  pre_g, beta1, beta2, beta3,
                  bias_correction1, bias_correction2, bias_correction3_sqrt,
                  lr, decay, eps, no_prox, grad_scale);  
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adan", &adan, "Adan optimized CUDA implementation.");
}
