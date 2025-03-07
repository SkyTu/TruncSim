/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-11-28 18:37:52
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-12-14 20:14:09
 * @FilePath: /txy/simulator/QPyTorch/qtorch/quant/quant_cuda/quant_cuda.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <torch/torch.h>
#include "quant_cuda.h"
#include <tuple>

using namespace at;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

Tensor fixed_point_quantize_nearest(Tensor a, int wl, int fl, bool use_clamp, bool symmetric)
{
  CHECK_INPUT(a);
  return fixed_point_quantize_nearest_cuda(a, wl, fl, use_clamp, symmetric);
}


Tensor fixed_point_quantize_prob_error_float(Tensor a, int wl, int bitlength, int fl, int trunc_type, bool clamp, bool trunc){
 
  CHECK_INPUT(a);
  
  return fixed_point_quantize_prob_error_cuda(a, wl, bitlength, fl, trunc_type, clamp, trunc, 0);
}


Tensor fixed_point_quantize_prob_error_double(Tensor a, int wl, int bitlength, int fl, int trunc_type, bool clamp, bool trunc){
 
  CHECK_INPUT(a);
  
  return fixed_point_quantize_prob_error_cuda(a, wl, bitlength, fl, trunc_type, clamp, trunc, 1);
}

std::tuple<Tensor, Tensor> fixed_point_quantize_nearest_mask(Tensor a, int wl, int fl,
                                                              bool symmetric)
{
  CHECK_INPUT(a);
  return fixed_point_quantize_nearest_mask_cuda(a, wl, fl, symmetric);
}

Tensor block_quantize_nearest(Tensor a, int wl, int dim)
{
  CHECK_INPUT(a);
  return block_quantize_nearest_cuda(a, wl, dim);
}

Tensor block_quantize_sim_nearest(Tensor a, int wl)
{
  CHECK_INPUT(a);
  return block_quantize_sim_nearest_cuda(a, wl);
}

Tensor float_quantize_nearest(Tensor a, int man_bits, int exp_bits)
{
  CHECK_INPUT(a);
  return float_quantize_nearest_cuda(a, man_bits, exp_bits);
}

Tensor fixed_point_quantize_stochastic(Tensor a, int wl, int fl, bool use_clamp, bool symmetric)
{
  CHECK_INPUT(a);
  return fixed_point_quantize_stochastic_cuda(a, wl, fl, use_clamp, symmetric);
}

std::tuple<Tensor, Tensor> fixed_point_quantize_stochastic_mask(Tensor a, int wl, int fl,
                                                                bool symmetric)
{
  CHECK_INPUT(a);
  return fixed_point_quantize_stochastic_mask_cuda(a, wl, fl, symmetric);
}

Tensor block_quantize_stochastic(Tensor a, int wl, int dim)
{
  CHECK_INPUT(a);
  return block_quantize_stochastic_cuda(a, wl, dim);
}

Tensor block_quantize_sim_stochastic(Tensor a, int wl)
{
  CHECK_INPUT(a);
  return block_quantize_sim_stochastic_cuda(a, wl);
}

Tensor float_quantize_stochastic(Tensor a, int man_bits, int exp_bits)
{
  CHECK_INPUT(a);
  return float_quantize_stochastic_cuda(a, man_bits, exp_bits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("fixed_point_quantize_prob_error_float", &fixed_point_quantize_prob_error_float, "Fixed Point Number Probabilistic Quantization with Error (CUDA)");
  m.def("fixed_point_quantize_prob_error_double", &fixed_point_quantize_prob_error_double, "Fixed Point Number Probabilistic Quantization with Error (CUDA)");
  m.def("fixed_point_quantize_stochastic", &fixed_point_quantize_stochastic, "Fixed Point Number Stochastic Quantization (CUDA)");
  m.def("fixed_point_quantize_stochastic_mask", &fixed_point_quantize_stochastic_mask, "Fixed Point Number Stochastic Quantization (CUDA)");
  m.def("block_quantize_stochastic", &block_quantize_stochastic, "Block Floating Point Number Stochastic Quantization (CUDA)");
  m.def("block_quantize_sim_stochastic", &block_quantize_sim_stochastic, "Block Floating Point Number Stochastic Quantization (CUDA)");
  m.def("float_quantize_stochastic", &float_quantize_stochastic, "Low-Bitwidth Floating Point Number Stochastic Quantization (CUDA)");
  m.def("fixed_point_quantize_nearest", &fixed_point_quantize_nearest, "Fixed Point Number Nearest Neighbor Quantization (CUDA)");
  m.def("fixed_point_quantize_nearest_mask", &fixed_point_quantize_nearest_mask, "Fixed Point Number Nearest Neighbor Quantization (CUDA)");
  m.def("block_quantize_nearest", &block_quantize_nearest, "Block Floating Point Number Nearest Neighbor Quantization (CUDA)");
  m.def("block_quantize_sim_nearest", &block_quantize_sim_nearest, "Block Floating Point Number Stochastic Quantization (CUDA)");
  m.def("float_quantize_nearest", &float_quantize_nearest, "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization (CUDA)");
}
