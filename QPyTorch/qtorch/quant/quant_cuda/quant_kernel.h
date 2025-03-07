#include <stdint.h>

__global__ void fixed_point_quantize_kernel_stochastic(float *__restrict__ a,
                                                       float *__restrict__ r,
                                                       float *o, int size,
                                                       int sigma, bool clamp,
                                                       float t_min, float t_max);

__global__ void fixed_point_quantize_kernel_nearest(float *__restrict__ a,
                                                    float *o, int size,
                                                    int sigma, bool clamp,
                                                    float t_min, float t_max);


// __global__ void fixed_point_quantize_kernel_prob_error(T *__restrict__ a,
//                                                     float *__restrict__ r,
//                                                     T *o, int size,
//                                                     int sigma, bool clamp,
//                                                     T t_min, T t_max,
//                                                     int wl, int bitlength, int fl, int trunc_type, bool trunc
//                                                     );

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


__global__ void fixed_point_quantize_kernel_mask_stochastic(float *__restrict__ a,
                                                            float *__restrict__ r,
                                                            float *o, uint8_t *mask,
                                                            int size, int sigma,
                                                            float t_min, float t_max);

__global__ void fixed_point_quantize_kernel_mask_nearest(float *__restrict__ a,
                                                         float *o, uint8_t *mask,
                                                         int size, int sigma,
                                                         float t_min, float t_max);

__global__ void float_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r,
                                        float *o, int size,
                                        int man_bits, int exp_bits);

__global__ void float_kernel_nearest(float *__restrict__ a,
                                     float *o, int size,
                                     int man_bits, int exp_bits);

__global__ void block_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r,
                                        float *o, int size,
                                        float *max_entry,
                                        int man_bits);

__global__ void block_kernel_nearest(float *__restrict__ a,
                                     float *o, int size,
                                     float *max_entry,
                                     int man_bits);

__global__ void block_kernel_sim_stochastic(float *__restrict__ a,
                                            float *__restrict__ r,
                                            float *o, int size,
                                            float *max_entry,
                                            int wl);

__global__ void block_kernel_sim_nearest(float *__restrict__ a,
                                         float *o, int size,
                                         float *max_entry,
                                         int wl);
