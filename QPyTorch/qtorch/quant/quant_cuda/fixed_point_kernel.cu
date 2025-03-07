#include "quant_kernel.h"
#include "sim_helper.cu"
#include <random>



template <typename T>
__device__ __forceinline__ float prob_gen(T prob, float value){
  return (value < prob) ? 1.0 : 0.0;
}

// template <typename T>
// __device__ __forceinline__ T trunc_fixed(T a, int sigma){
//   a = ldexp(a, -sigma);
//   a = std::trunc(a);
//   a = ldexp(a, sigma);
//   return a;
// }

// template <typename T>
// __device__ void convert_to_fixed_point_with_trunc(T & a, int wl, int fl, T pp, int pv, T np, int nv, float random_values)
// {
//   if (a == 0){
//     return;
//   }
//   if(a < 0){
//     a = a+prob_gen(np, random_values)*T(nv);
//   }
//   else{
//     a = a+prob_gen(pp, random_values)*T(pv) ;
//   }
// }

template <typename T>
__device__ __forceinline__ T clamp_helper(T a, T min, T max) {
  if (a > max) return max;
  else if (a < min) return min;
  else return a;
}

template <typename T>
__device__ __forceinline__ T clamp_mask_helper(T a, T min, T max, uint8_t* mask) {
  if (a > max) {
    *mask = 1;
    return max;
  } else if (a < min) {
    *mask = 1;
    return min;
  }
  *mask = 0;
  return a;
}

// quantize an array of real numbers into fixed point with word length [wl] and [fl] fractional bits
// 2**-[sigma] is the smallest unit of the fixed point representation. Stochastic Rounding with r.
__global__ void fixed_point_quantize_kernel_stochastic(float* __restrict__ a,
                                                       float* __restrict__ r,
                                                       float* o, int size,
                                                       int sigma, bool use_clamp,
                                                       float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = round(a[index], r[index], sigma);
    if (use_clamp) {
      o[index] = clamp_helper(o[index], t_min, t_max);
    }
  }
}

// quantize an array of real numbers into fixed point with word length [wl] and [fl] fractional bits
// 2**-[sigma] is the smallest unit of the fixed point representation. Nearest Neighbor Rounding.
__global__ void fixed_point_quantize_kernel_nearest(float* __restrict__ a,
                                                    float* o, int size,
                                                    int sigma, bool use_clamp,
                                                    float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = nearest_round(a[index], sigma);
    if (use_clamp) {
      o[index] = clamp_helper(o[index], t_min, t_max);
    }
  }
}


__global__ void fixed_point_quantize_kernel_prob_error(float* __restrict__ a,
                                                      float *__restrict__ r1, float *__restrict__ r2,
                                                      float* o, int size,
                                                      int sigma, bool use_clamp,
                                                      float t_min, float t_max,
                                                      int wl, int bitlength, int fl, int trunc_type,
                                                      bool trunc) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    // if(trunc){
        o[index] = ldexp(a[index], -sigma);
        if (trunc_type == 0){
          o[index] = floor(o[index]);
        }
        if (trunc_type == 1){
          auto pr1 = o[index] - truncf(o[index]) +1 - truncf(o[index] - truncf(o[index])+1);
          o[index] = o[index] + prob_gen(pr1, r1[index]);
          o[index] = floor(o[index]);
        }

        if (trunc_type == 2){
          auto pr1 = o[index] - truncf(o[index]) +1 - truncf(o[index] - truncf(o[index])+1);
          auto pr2 =  o[index]>0 ? o[index] / ldexp(1, wl+sigma):- o[index] / ldexp(1, wl+sigma);
          pr1 =  o[index]  > 0 || pr1==0 ? pr1:1-pr1;

          int tmp = prob_gen(pr2, r1[index]);
          int tmp2 = prob_gen(pr1, r2[index]);

          bool positvie = o[index] > 0;
          o[index] = positvie ? o[index] + tmp2:o[index] - tmp2;
          o[index] = positvie ? o[index] - ldexp(1, wl-fl) * tmp:o[index] + ldexp(1, wl-fl) * tmp;

          o[index] = floor(o[index]);
        }

       

        // convert_to_fixed_point_with_trunc(o[index], wl, fl, pp, pv, np, nv, r[index]);
        o[index] = ldexp(o[index], sigma);        
    // }
    // if (use_clamp) {
    //   o[index] = clamp_helper(o[index], t_min, t_max);
    // }
  }

}


__global__ void fixed_point_quantize_kernel_prob_error(double* __restrict__ a,
                                                      float *__restrict__ r1, float *__restrict__ r2,
                                                      double* o, int size,
                                                      int sigma, bool use_clamp,
                                                      double t_min, double t_max,
                                                      int wl, int bitlength, int fl, int trunc_type,
                                                      bool trunc) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    // o[index] = trunc_fixed(a[index], sigma);
    // 如果是乘法，则需要截断操作，截断后就有概率出现误差
    // if(trunc){
         o[index] = ldexp(a[index], -sigma);

        if (trunc_type ==1){
          auto pr1 = o[index] - truncf(o[index]) +1 - truncf(o[index] - truncf(o[index])+1);
          // pr1 =  o[index]  > 0 || pr1==0 ? pr1:1-pr1;
          o[index] = o[index] + prob_gen(pr1, r1[index]);
        }

        if (trunc_type ==2){
          auto pr2 =  o[index]>0 ? o[index] / ldexp(1, wl+sigma):- o[index] / ldexp(1, wl+sigma);
          int tmp = prob_gen(pr2, r1[size+index]);
          auto pr1 = o[index] - truncf(o[index]) +1 - truncf(o[index] - truncf(o[index])+1);
          pr1 =  o[index]  > 0 || pr1==0 ? pr1:1-pr1;
          o[index] = o[index] > 0 ? o[index] - ldexp(1, wl-fl) * tmp:o[index] + ldexp(1, wl-fl) * tmp;
          o[index] = o[index] > 0 ? o[index] + prob_gen(pr1, r1[index]):o[index] - prob_gen(pr1, r1[index]);
        }

        o[index] = floor(o[index]);

        // convert_to_fixed_point_with_trunc(o[index], wl, fl, pp, pv, np, nv, r[index]);
        o[index] = ldexp(o[index], sigma);     
    // }
  }

}
__global__ void fixed_point_quantize_kernel_mask_stochastic(float* __restrict__ a,
                                                            float *__restrict__ r,
                                                            float* o, uint8_t* m,
                                                            int size, int sigma,
                                                            float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = round(a[index], r[index], sigma);
    o[index] = clamp_mask_helper(o[index], t_min, t_max, m+index);
  }
}

__global__ void fixed_point_quantize_kernel_mask_nearest(float* __restrict__ a,
                                                         float* o, uint8_t* m,
                                                         int size, int sigma,
                                                         float t_min, float t_max) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    o[index] = nearest_round(a[index], sigma);
    o[index] = clamp_mask_helper(o[index], t_min, t_max, m+index);
  }
}

__global__ void fixed_point_quantize_kernel_prob_error(float* __restrict__ a,
                                                    float *__restrict__ r1, float *__restrict__ r2,
                                                    float* o, int size,
                                                    int sigma, bool use_clamp,
                                                    float t_min, float t_max, int wl, int bitlength, int fl, int trunc_type,
                                                    bool trunc);

__global__ void fixed_point_quantize_kernel_prob_error(double* __restrict__ a,
                                                    float *__restrict__ r1, float *__restrict__ r2,
                                                    double* o, int size,
                                                    int sigma, bool use_clamp,
                                                    double t_min, double t_max, int wl, int bitlength, int fl, int trunc_type,
                                                    bool trunc);