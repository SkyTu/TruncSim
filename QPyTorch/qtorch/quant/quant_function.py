import torch
from .. import Number, FixedPoint, BlockFloatingPoint, FloatingPoint, FixedPointWithProbError
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np

import os
from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_quant():
    from torch.utils.cpp_extension import load
    current_path = os.path.dirname(os.path.realpath(__file__))
    quant_cpu = load(
        name="quant_cpu",
        sources=[
            os.path.join(current_path, "quant_cpu/quant_cpu.cpp"),
            os.path.join(current_path, "quant_cpu/bit_helper.cpp"),
            os.path.join(current_path, "quant_cpu/sim_helper.cpp"),
        ],
        extra_ldflags=["-L /usr/share/lib", "-lgmp", "-lgmpxx"]
    )

    if torch.cuda.is_available():
        quant_cuda = load(
            name="quant_cuda",
            sources=[
                os.path.join(current_path, "quant_cuda/quant_cuda.cpp"),
                os.path.join(current_path, "quant_cuda/bit_helper.cu"),
                os.path.join(current_path, "quant_cuda/sim_helper.cu"),
                os.path.join(current_path, "quant_cuda/block_kernel.cu"),
                os.path.join(current_path, "quant_cuda/float_kernel.cu"),
                os.path.join(current_path, "quant_cuda/fixed_point_kernel.cu"),
                os.path.join(current_path, "quant_cuda/quant.cu"),
            ],
        )
    else:
        quant_cuda = quant_cpu
    return quant_cpu, quant_cuda

__all__ = ["fixed_point_quantize", "block_quantize", "float_quantize", "quantizer"]


def assert_wl_fl(wl, fl, stage=""):
    if wl == -1 and fl != -1:
        raise ValueError("fixed point {} wl {}, fl {}".format(stage, wl, fl))


def get_module(x):
    quant_cpu, quant_cuda = get_quant()
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu
    else:
        if x.tensor.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu
    return quant_module


def quantizer(
    forward_number=None,
    backward_number=None,
    forward_rounding="stochastic",
    backward_rounding="stochastic",
    clamping_grad_zero=False,
    backward_hooks=[],
):
    """
    Creates a quantization function to support quantizing forward and backward process differently.

    Args:
        - :param: forward_number (qtorch.Number, optional) : the number format used for forward quantization.
                  if is None, the quantization would be a identity mapping.
        - :param: backward_number (qtorch.Number, optional) : the number format used for backward quantization.
                  if is None, the quantization would be a identity mapping.
        - :param: forward_rounding (string) : rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")
        - :param: backward_rounding (string) : rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")
        - :param: clamping_grad_zero (bool) : zero out the gradient of numbers that are being clamped during forward propagation.
                  currently requires forward_number to be a fixed point number.
        - :param: backward_hooks (iterable) : iterable of functions that will be applied to gradients before backward quantization.
                  For example, this can be used to support custom scaling.

    Returns:
        A quantization function as specified (torch.Tensor -> torch.Tensor)
    """

    for rounding in [forward_rounding, backward_rounding]:
        assert rounding in ["stochastic", "nearest", "prob_error"], "invalid rounding type {:s}".format(rounding)
    for num in [forward_number, backward_number]:
        if num != None:
            assert isinstance(num, Number)

    if clamping_grad_zero == False:
        if forward_rounding == "nearest":
            if type(forward_number) == BlockFloatingPoint:
                forward_quant = lambda x, quant_module: quant_module.block_quantize_nearest(
                    x, forward_number.wl, forward_number.dim
                )
            elif type(forward_number) == FixedPoint:
                forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_nearest(
                    x, forward_number.wl, forward_number.fl, forward_number.clamp, forward_number.symmetric,
                )
            elif type(forward_number) == FloatingPoint:
                forward_quant = lambda x, quant_module: quant_module.float_quantize_nearest(
                    x, forward_number.man, forward_number.exp
                )
        elif forward_rounding == "stochastic":
            if type(forward_number) == BlockFloatingPoint:
                forward_quant = lambda x, quant_module: quant_module.block_quantize_stochastic(
                    x, forward_number.wl, forward_number.dim
                )
            elif type(forward_number) == FixedPoint:
                forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_stochastic(
                    x, forward_number.wl, forward_number.fl, forward_number.clamp, forward_number.symmetric,
                )
            elif type(forward_number) == FloatingPoint:
                forward_quant = lambda x, quant_module: quant_module.float_quantize_stochastic(
                    x, forward_number.man, forward_number.exp
                )
        # elif forward_rounding == "prob_error":
        #     if type(forward_number) == FixedPointWithProbError:
        #         if forward_number!=None:
        #             tensor_type = forward_number.tensor_type
        #         if(tensor_type == 32):
        #             forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_prob_error_float(
        #                 x, forward_number.wl, forward_number.fl, forward_number.pp, forward_number.pv, forward_number.np, forward_number.nv, forward_number.clamp, forward_number.trunc,
        #             )
        #         else:
        #             forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_prob_error_double(
        #                 x, forward_number.wl, forward_number.fl, forward_number.pp, forward_number.pv, forward_number.np, forward_number.nv, forward_number.clamp, forward_number.trunc,
        #             )
    else:
        if type(forward_number) == FixedPoint or forward_number == None:
            assert (
                forward_number == None or forward_number.clamp == True
            ), "must use clamping if zeroing out clamped gradient"
            if forward_rounding == "nearest":
                forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_nearest_mask(
                    x, forward_number.wl, forward_number.fl, forward_number.symmetric
                )
            elif forward_rounding == "stochastic":
                forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_stochastic_mask(
                    x, forward_number.wl, forward_number.fl, forward_number.symmetric
                )
            # elif forward_rounding == "prob_error":
            #     if forward_number!=None:
            #         tensor_type = forward_number.tensor_type
            #     if(tensor_type == 32):
            #         forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_prob_error_float(
            #             x, forward_number.wl, forward_number.fl, forward_number.pp, forward_number.pv, forward_number.np, forward_number.nv, forward_number.clamp, forward_number.symmetric,
            #         )
            #     else:
            #         forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_prob_error_double(
            #             x, forward_number.wl, forward_number.fl, forward_number.pp, forward_number.pv, forward_number.np, forward_number.nv, forward_number.clamp, forward_number.symmetric,
            #         )
        else:
            raise ValueError("zeroing clamping gradient only support fixed point.")

    if backward_rounding == "nearest":
        if type(backward_number) == BlockFloatingPoint:
            backward_quant = lambda a, quant_module: quant_module.block_quantize_nearest(
                a, backward_number.wl, backward_number.dim
            )
        elif type(backward_number) == FixedPoint:
            backward_quant = lambda a, quant_module: quant_module.fixed_point_quantize_nearest(
                a, backward_number.wl, backward_number.fl, backward_number.clamp, backward_number.symmetric,
            )
        elif type(backward_number) == FloatingPoint:
            backward_quant = lambda a, quant_module: quant_module.float_quantize_nearest(
                a, backward_number.man, backward_number.exp
            )
    elif backward_rounding == "stochastic":
        if type(backward_number) == BlockFloatingPoint:
            backward_quant = lambda a, quant_module: quant_module.block_quantize_stochastic(
                a, backward_number.wl, backward_number.dim
            )
        elif type(backward_number) == FixedPoint:
            backward_quant = lambda a, quant_module: quant_module.fixed_point_quantize_stochastic(
                a, backward_number.wl, backward_number.fl, backward_number.clamp, backward_number.symmetric,
            )
        elif type(backward_number) == FloatingPoint:
            backward_quant = lambda a, quant_module: quant_module.float_quantize_stochastic(
                a, backward_number.man, backward_number.exp
            )
    # elif backward_rounding == "prob_error":
    #     if type(backward_number) == FixedPointWithProbError:
    #         if backward_number!=None:
    #             tensor_type = backward_number.tensor_type
    #         if(tensor_type == 32):
    #             backward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_prob_error_float(
    #                 x, backward_number.wl, backward_number.fl, backward_number.pp, backward_number.pv, backward_number.np, backward_number.nv, backward_number.clamp, backward_number.symmetric,
    #             )
    #         else:
    #             backward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_prob_error_double(
    #                 x, backward_number.wl, backward_number.fl, backward_number.pp, backward_number.pv, backward_number.np, backward_number.nv, backward_number.clamp, backward_number.symmetric,
    #             )

    if clamping_grad_zero == False:

        class Rounding(torch.autograd.Function):
            @staticmethod
            def forward(self, x):
                if forward_number == None:
                    return x

                quant_module = get_module(x)
                if isinstance(x, torch.Tensor):
                    out = forward_quant(x.contiguous(), quant_module)
                else:
                    out = forward_quant(x.tensor.contiguous(), quant_module)

                return out

            @staticmethod
            def backward(self, grad_output):
                if self.needs_input_grad[0]:
                    if backward_number == None:
                        grad_input = grad_output
                    else:
                        quant_module = get_module(grad_output)
                        grad_input = backward_quant(grad_output.contiguous(), quant_module)
                else:
                    grad_input = None

                return grad_input

    else:

        class Rounding(torch.autograd.Function):
            @staticmethod
            def forward(self, x):
                if forward_number == None:
                    self.mask = torch.zeros_like(x).bool()
                    return x
                else:
                    quant_module = get_module(x)
                    out, mask = forward_quant(x.contiguous(), quant_module)
                    self.mask = mask

                return out

            @staticmethod
            def backward(self, grad_output):
                if self.needs_input_grad[0]:
                    if backward_number == None:
                        grad_input = grad_output
                    else:
                        quant_module = get_module(grad_output)
                        # grad_output = grad_output.contiguous().masked_fill_(self.mask, 0)
                        for f in backward_hooks:
                            grad_output = f(grad_output)
                        grad_input = backward_quant(grad_output.contiguous(), quant_module).masked_fill(
                            self.mask.bool(), 0
                        )
                else:
                    grad_input = None

                return grad_input

    return Rounding.apply


def fixed_point_quantize(x, wl, bitlength, fl, trunc_type, clamp=True, symmetric=False, rounding="prob_error", trunc = False):
    """
    Quantize a single precision Floating Point into low-precision Fixed Point

    Args:
        - :param: `x` (torch.Tensor) :  the single precision number to be quantized
        - :param: `wl` (int) : word length of the fixed point number being simulated
        - :param: `fl` (int) : fractional length of the fixed point number being simulated
        - :param: `clamp` (bool, optional) : clamp input numbers into representable range. if false,
                  the quantization will only simulate the effect on precision
        - :param: `symmetric` (bool, optional) : discard the minimum representable number to make the representable
                  range symmetric
        - :param: `rounding` (string) : rounding mode, \"stochastic\" or \"nearest\" or \"prob_error" default: \"prob_error\"
        - :param: `pp` (float) : probability of positive error
        - :param: `pv` (int) : positive error value
        - :param: `np` (float) : probability of negative error
        - :param: `nv` (int) : negative error value

    Returns:
        - a quantized low-precision block floating point number (torch.Tensor)
    """
    assert isinstance(x, torch.Tensor)
    assert rounding in ["stochastic", "nearest", "prob_error"]
    assert_wl_fl(wl, fl)
    quant_module = get_module(x)
    if rounding == "nearest":
        if x.dtype == torch.int64 or x.dtype == torch.int32:
            return x
        out = quant_module.fixed_point_quantize_nearest(x.contiguous(), wl, fl, clamp, symmetric)
    elif rounding == "stochastic":
        out = quant_module.fixed_point_quantize_stochastic(x.contiguous(), wl, fl, clamp, symmetric)
    elif rounding == "prob_error":
        if x.dtype == torch.float32:
            out = quant_module.fixed_point_quantize_prob_error_float(x.contiguous(), wl, bitlength, fl, trunc_type, clamp, trunc)
        elif x.dtype == torch.float64:
            out = quant_module.fixed_point_quantize_prob_error_double(x.contiguous(), wl, bitlength, fl, trunc_type, clamp, trunc)
        elif x.dtype == torch.int64 or x.dtype == torch.int32:
            return x
        else:
            raise ValueError("prob_error rounding only support float32 and float64")
    return out


def block_quantize(x, wl, dim=-1, rounding="stochastic"):
    """
    Quantize a single precision Floating Point into low-precision Block Floating Point

    Args:
        - :param: `x` (torch.Tensor) :  the single precision number to be quantized
        - :param: `wl` (int) : word length of the block floating point number being simulated
        - :param: `rounding` (string) : rounding mode, \"stochastic\" or \"nearest\"

    Returns:
        - a quantized low-precision block floating point number (torch.Tensor)
    """
    assert isinstance(x, torch.Tensor), "x is not a single precision Floating Point Tensor"
    assert rounding in ["stochastic", "nearest"], "invalid rounding mode, {}".format(rounding)
    quant_module = get_module(x)
    if rounding == "nearest":
        out = quant_module.block_quantize_nearest(x.contiguous(), wl, dim)
    elif rounding == "stochastic":
        out = quant_module.block_quantize_stochastic(x.contiguous(), wl, dim)
    return out


def float_quantize(x, exp, man, rounding="stochastic"):
    """
    Quantize a single precision Floating Point into low-precision Floating Point

    Args:
        - :attr: `x` (torch.Tensor) : the single precision number(torch.Tensor) to be quantized
        - :attr: `exp` (int) : number of bits allocated for exponent
        - :attr: `man` (int) : number of bits allocated for mantissa, not counting the virtual bit
        - :attr: `rounding` (string) : rounding mode, \"stochastic\" or \"nearest\"

    Returns:
        - a quantized low-precision floating point number (torch.Tensor)
    """
    assert isinstance(x, torch.Tensor), "x is not a single precision Floating Point Tensor"
    assert rounding in ["stochastic", "nearest"], "invalid rounding mode, {}".format(rounding)
    quant_module = get_module(x)
    if rounding == "nearest":
        out = quant_module.float_quantize_nearest(x.contiguous(), man, exp)
    elif rounding == "stochastic":
        out = quant_module.float_quantize_stochastic(x.contiguous(), man, exp)
    return out
