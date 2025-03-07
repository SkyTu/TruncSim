/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-11-28 18:37:52
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-12-12 10:10:23
 * @FilePath: /txy/simulator/QPyTorch/qtorch/quant/quant_cpu/sim_helper.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "quant_cpu.h"
#include <math.h>
#include <stdint.h>
#ifndef SIM_HELPER_CPP_
#define SIM_HELPER_CPP_



float round(float a, float r, int sigma)
{
  a = ldexp(a, -sigma);
  a = nearbyint(a + r - 0.5);
  // a = floor(a + r);
  a = ldexp(a, sigma);
  return a;
}

#endif

