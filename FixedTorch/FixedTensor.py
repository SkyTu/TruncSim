import sys
import os
quantPath = os.path.relpath("../FixedTensor")
sys.path.append(quantPath)
qPath = os.path.relpath("../QPyTorch")
sys.path.append(qPath)
import torch
from qtorch.quant import fixed_point_quantize

import Parameter
from functools import reduce
from torch import Tensor
import math
param = Parameter.Parameter()

def approx_exp(input, iter = 9):
    with torch.no_grad():
        n = 1 << iter
        tmp = FixedTensor(torch.div(input,  n))
        a = FixedTensor(1 + tmp).quant()
        
        for i in range(0, iter):
            # a = a * a
            a.mul_(a)
            a = FixedTensor(a).quant()
        a[input<=-(math.log(2**param.fl))] = 0    
        return a  


def inversqrt(input):
    with torch.no_grad():
        exp = torch.log2(input)
        exp = exp.trunc()+2
        mini_input = FixedTensor(torch.div(input, 2**exp)).quant()
        inversqrt_exp = FixedTensor(torch.exp2(-0.5*exp)).quant()
        inversqrt_mini_input = 3.14736 + mini_input * FixedTensor(4.63887 * mini_input - 5.77789).quant()
        res = FixedTensor(inversqrt_exp*inversqrt_mini_input).quant()
        return res
    
def sqrt(input):
    with torch.no_grad():
        exp = torch.log2(input)
        exp = exp.trunc()+1
        g_0 = FixedTensor(input).div(torch.exp2(0.5*exp))
        h_0 = FixedTensor(1/torch.exp2(0.5*exp)).quant()*0.5
        gh_0 = g_0*h_0
        g= g_0
        h = h_0
        gh =gh_0
        for i in range(1, 4):
            r = 1.5 - gh
            g = g*r
            h = h*r
            gh = g*h
        r = 1.5 -gh
        h = h*r
        H = h*h*4
        H= H*input
        H = 3-H
        H = FixedTensor(h*H).quant()
        res = H*input
        res[input<=0] = 0
        return FixedTensor(res)
    
def sigma_exp(input, lut_in_fl=None, lut_out_fl=None):
    """SIGMA-style exp via LUT, simulated.
    Errors come from: (1) input quantization at LUT-index precision,
    (2) output quantization at LUT-entry precision. The "lookup" itself
    returns the true value at the quantized index. Underflow region
    (x <= -log(2^fl)) is clipped to 0 to match SIGMA's bounded LUT range.
    """
    with torch.no_grad():
        in_fl = param.fl if lut_in_fl is None else lut_in_fl
        out_fl = param.fl if lut_out_fl is None else lut_out_fl
        x_q = FixedTensor(input).quant(fl=in_fl)
        result = torch.exp(x_q)
        result[x_q <= -math.log(2**param.fl)] = 0
        return FixedTensor(result).quant(fl=out_fl)


def sigma_recip(input, lut_in_fl=None, lut_out_fl=None):
    """SIGMA-style 1/x via LUT, simulated.
    Same error structure as sigma_exp: input quant + true reciprocal +
    output quant. For softmax usage the input is sum(exp) >= 1, so no
    div-by-zero concerns.
    """
    with torch.no_grad():
        in_fl = param.fl if lut_in_fl is None else lut_in_fl
        out_fl = param.fl if lut_out_fl is None else lut_out_fl
        y_q = FixedTensor(input).quant(fl=in_fl)
        # Floor at LSB to be safe against rare zero-sum cases.
        y_q = torch.where(y_q.abs() < 2.0**(-in_fl),
                          torch.full_like(y_q, 2.0**(-in_fl)),
                          y_q)
        result = 1.0 / y_q
        return FixedTensor(result).quant(fl=out_fl)


def approx_div(input1, input2, iter = 10):
    with torch.no_grad():
        
        tmp_input = torch.abs(input2)
        z = FixedTensor(3 * approx_exp(1-2*tmp_input) + 0.003)
        for i in range(iter):
            z = FixedTensor(2*z- z*tmp_input*z)
        z = FixedTensor(z*torch.sign(input2))
        res = z*input1
        return res
    
def two_power(n):
    if isinstance(n, int) and n < 31:
        return 2**n
    
def PreOpL(op, items):
    """
    Uses algorithm from SecureSCM WP9 deliverable.
    
    op must be a binary function that outputs a new register
    """
    k = len(items)
    logk = int(math.ceil(math.log(k,2)))
    kmax = 2**logk
    output = list(items)
    for i in range(logk):
        for j in range(kmax//(2**(i+1))):
            y = two_power(i) + j*two_power(i+1) - 1
            for z in range(1, 2**i+1):
                if y+z < k:
                    output[y+z] = op(output[y], output[y+z], j != 0)
    return output
def log(input, b):
    with torch.no_grad():
        logb_2 = math.log(2, b)
        exp = torch.log2(input)
        exp = exp.trunc()+1
        mini_input = FixedTensor(torch.div(input,2**exp)).quant()
        pre_mults = PreOpL(lambda a,b,_: a * b,
                                        [mini_input] * 4)        
        p_2524 = [-2.05466671951, -8.8626599391,
          +6.10585199015, +4.81147460989]  
        q_2524 = [+0.353553425277, +4.54517087629,
          +6.42784209029, +1]
        P = p_2524[0]
        Q =  q_2524[0]
        for i in range(3):
            P += pre_mults[i] * p_2524[i+1]
            Q += pre_mults[i] * q_2524[i+1]
        res = P.div(Q)
        return (res+exp)*logb_2
    
class LogFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, b = 2):
        res = log(input, b)
        ctx.save_for_backward(input, Tensor([b]))
        return res

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_b = None
        input, b,  = ctx.saved_tensors
        if not isinstance(grad_output, FixedTensor):
            grad_output = FixedTensor(grad_output).quant()
        if ctx.needs_input_grad[0]:
            grad_input = grad_output*approx_div(1, input*math.log(math.e, int(b[0]))) 
        return grad_input, grad_b 

class InverSqrtFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        res = inversqrt(input)
        ctx.save_for_backward(res)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        res, = ctx.saved_tensors
        if not isinstance(grad_output, FixedTensor):
            grad_output = FixedTensor(grad_output).quant()
        if ctx.needs_input_grad[0]:
            grad_input = res *grad_output *res * res * -0.5#FixedTensor(-0.5*FixedTensor(res*FixedTensor(res*FixedTensor(res*grad_output).quant()).quant()).quant()).quant()
        return grad_input



class SqrtFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        res = sqrt(input)
        ctx.save_for_backward(res)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        res, = ctx.saved_tensors
        if not isinstance(grad_output, FixedTensor):
            grad_output = FixedTensor(grad_output).quant()
        if ctx.needs_input_grad[0]:
            grad_input = FixedTensor(0.5*FixedTensor(approx_div(1, res)))
        return grad_input
    
class ExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, iter=8):
        res = approx_exp(input, iter)
        ctx.save_for_backward(res)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if not isinstance(grad_output, FixedTensor):
            grad_output = FixedTensor(grad_output).quant()
        a, = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * a
        return grad_input, None

class DivFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, iter = 10):
        if isinstance(input2, float) or isinstance(input2, int):
            input2 = torch.tensor(input2)
        tmp_input = torch.abs(input2)
        z = FixedTensor(3 * approx_exp(1-2*tmp_input)+0.003)
        for i in range(iter):
            z = FixedTensor(2*z - z*tmp_input*z)
            
        z = FixedTensor(z*torch.sign(input2))
        res = z*input1
        ctx.save_for_backward( res, z)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        grad_input1 = grad_input2 = None
        res, z = ctx.saved_tensors
        if not isinstance(grad_output, FixedTensor):
            grad_output = FixedTensor(grad_output).quant()
        if isinstance(res, tuple):
            res = res[0]
        if ctx.needs_input_grad[0]:
            grad_input1 =grad_output*z
        if ctx.needs_input_grad[1]:
            grad_input2 = grad_output*res*z   
        return grad_input1, grad_input2, None

class MeanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim = None, keepdim=False):
        if dim == None or len(input.shape)==1:
            stride = torch.numel(input)
        else:
            if isinstance(dim, int):
                dim = [dim]
            stride = reduce(lambda x, y: x * input.size()[y], dim, 1)
        tmp = torch.zeros_like(input)
        tmp[:] = 1/stride
        ctx.save_for_backward(tmp)
        res = FixedTensor(torch.mean(input, dim=dim, keepdim = keepdim)).quant()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_input1 = grad_input2 = None
        tmp, = ctx.saved_tensors
        if not isinstance(grad_output, FixedTensor):
            grad_output = FixedTensor(grad_output)
        if ctx.needs_input_grad[0]:
            grad_input = grad_output*tmp
        return grad_input, grad_input1, grad_input2

class VarFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim = None, keepdim=False,unbiased=False):
        if dim == None or len(input.shape)==1:
            stride = torch.numel(input)
        else:
            if isinstance(dim, int):
                dim = [dim]
            stride = reduce(lambda x, y: x * input.size()[y], dim, 1)
        # tmp = torch.zeros_like(input)
        # tmp[:] = 1/stride
        # ctx.save_for_backward(tmp)
        
        mean = FixedTensor(torch.mean(input, dim=dim, keepdim = keepdim)).quant()
        dmean = input - mean
        var = FixedTensor(dmean**2).quant()
        if unbiased:
            res = FixedTensor(torch.sum(var, dim=dim, keepdim = keepdim) / (stride)).quant()
        else:
            res = FixedTensor(torch.sum(var, dim=dim, keepdim = keepdim) / (stride - 1)).quant()
        ctx.save_for_backward(res, dmean, stride)
        return res, None, None, None

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if not isinstance(grad_output, FixedTensor):
            grad_output = FixedTensor(grad_output).quant()
        std, dmean, stride = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            tmp = approx_div(grad_output, std)
            tmp1 = FixedTensor(tmp*dmean).quant()
            grad_input = FixedTensor(tmp1/(stride-1)).quant()
        return grad_input
    
class StdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim = None, keepdim=False,unbiased=False):
        if dim == None or len(input.shape)==1:
            stride = torch.numel(input)
        else:
            if isinstance(dim, int):
                dim = [dim]
            stride = reduce(lambda x, y: x * input.size()[y], dim, 1)
        # tmp = torch.zeros_like(input)
        # tmp[:] = 1/stride
        # ctx.save_for_backward(tmp)
        
        mean = FixedTensor(torch.mean(input, dim=dim, keepdim = keepdim)).quant()
        dmean = input - mean
        std = FixedTensor(dmean**2).quant()
        
        if unbiased:
            std = FixedTensor(torch.sum(std, dim=dim, keepdim = keepdim) / (stride)).quant()
        else:
            std = FixedTensor(torch.sum(std, dim=dim, keepdim = keepdim) / (stride - 1)).quant()
        res = sqrt(std)
        ctx.save_for_backward(std, dmean, stride)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if not isinstance(grad_output, FixedTensor):
            grad_output = FixedTensor(grad_output).quant()
        std, dmean, stride = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            tmp = approx_div(grad_output, std)
            tmp1 = FixedTensor(tmp*dmean).quant()
            grad_input = FixedTensor(tmp1/(stride-1)).quant()
        return grad_input


class MulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        if isinstance(input2, float) or isinstance(input2, int):
            tmp_input2 = torch.tensor(input2)
            ctx.save_for_backward(input1, tmp_input2)
        else:
            ctx.save_for_backward(input1, input2)
        res = FixedTensor(torch.mul(input1, input2)).quant()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        grad_input1 = grad_input2 = None
        grad_output = FixedTensor(grad_output).quant()
        input1, input2 = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input1 = FixedTensor(torch.mul(grad_output, input2)).quant()
        if ctx.needs_input_grad[1]:
            grad_input2 = FixedTensor(torch.mul(grad_output, input1)).quant()
            
        return grad_input1, grad_input2
    
def _softmax_backward(ctx, grad_output):
    """Standard softmax-Jacobian backward, shared across all forward variants.
    The ReLU-trick has a different exact gradient, but in MPC training the
    standard softmax gradient is commonly used as an approximation.
    """
    res, dim = ctx.saved_tensors
    grad_input = None
    dim = int(dim[0])
    if ctx.needs_input_grad[0]:
        if not isinstance(grad_output, FixedTensor):
            grad_output = FixedTensor(grad_output).quant()
        inter1 = torch.sum(FixedTensor(grad_output*res).quant(), dim=dim, keepdim=True)
        inter2 = grad_output - inter1
        grad_input = FixedTensor(res*inter2).quant()
    return grad_input, None


def _logsoftmax_backward(ctx, grad_output):
    softmax_value, dim = ctx.saved_tensors
    grad_input = None
    dim = int(dim[0])
    if ctx.needs_input_grad[0]:
        if not isinstance(grad_output, FixedTensor):
            grad_output = FixedTensor(grad_output).quant()
        inter1 = FixedTensor(torch.sum(grad_output, dim=dim, keepdim=True))
        inter2 = inter1*softmax_value
        grad_input = FixedTensor(inter2 - grad_output)
    return grad_input, None


class PiranhaSubMaxSoftMax(torch.autograd.Function):
    """Piranha softmax (subtract-max variant): the protocol used in USENIX'22 Piranha.
       exp via iterated-squaring limit approximation, division via Goldschmidt.
    """
    @staticmethod
    def forward(ctx, input, dim=-1):
        input_max = torch.max(input, dim=dim, keepdim=True).values
        intermdiate = input - input_max
        intermdiate_exp = approx_exp(intermdiate)
        res = approx_div(intermdiate_exp, torch.sum(intermdiate_exp, dim=dim, keepdim=True))
        ctx.save_for_backward(res, Tensor([dim]))
        return res

    backward = staticmethod(_softmax_backward)


class PiranhaReluSoftMax(torch.autograd.Function):
    """Piranha softmax (ReLU-trick variant): softmax(x) ~ ReLU(x) / sum(ReLU(x)).
       No exp needed; division via Goldschmidt. Matches SecureML / Piranha §5.
    """
    @staticmethod
    def forward(ctx, input, dim=-1):
        relu_x = FixedTensor(torch.clamp(input, min=0)).quant()
        sum_relu = FixedTensor(torch.sum(relu_x, dim=dim, keepdim=True)).quant()
        # Floor at LSB to avoid 0/0 when an entire row is non-positive.
        sum_relu = FixedTensor(torch.clamp_min(sum_relu, 2.0**(-param.fl))).quant()
        res = approx_div(relu_x, sum_relu)
        ctx.save_for_backward(res, Tensor([dim]))
        return res

    backward = staticmethod(_softmax_backward)


class SigmaSoftMax(torch.autograd.Function):
    """SIGMA softmax (PETS'24, simplified): subtract-max + LUT-modeled exp + LUT-modeled 1/sum.
       Per the simplified model, LUT lookup = quant(input) -> true op -> quant(output).
    """
    @staticmethod
    def forward(ctx, input, dim=-1):
        input_max = torch.max(input, dim=dim, keepdim=True).values
        intermdiate = input - input_max
        intermdiate_exp = sigma_exp(intermdiate)
        sum_q = FixedTensor(torch.sum(intermdiate_exp, dim=dim, keepdim=True)).quant()
        recip = sigma_recip(sum_q)
        res = FixedTensor(intermdiate_exp * recip).quant()
        ctx.save_for_backward(res, Tensor([dim]))
        return res

    backward = staticmethod(_softmax_backward)


class PiranhaSubMaxLogSoftMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=-1):
        input_max = torch.max(input, dim=dim, keepdim=True).values
        intermdiate = input - input_max
        intermdiate_exp = approx_exp(intermdiate)
        sum_ex = FixedTensor(torch.sum(intermdiate_exp, dim=dim, keepdim=True))
        logsum = log(sum_ex, math.e)
        res = intermdiate - logsum
        ctx.save_for_backward(approx_div(intermdiate_exp, sum_ex), Tensor([dim]))
        return res

    backward = staticmethod(_logsoftmax_backward)


class PiranhaReluLogSoftMax(torch.autograd.Function):
    """log(ReLU-trick softmax). Floor before log to avoid log(0); the
       resulting gradient is approximated by the standard softmax-grad.
    """
    @staticmethod
    def forward(ctx, input, dim=-1):
        relu_x = FixedTensor(torch.clamp(input, min=0)).quant()
        sum_relu = FixedTensor(torch.sum(relu_x, dim=dim, keepdim=True)).quant()
        sum_relu = FixedTensor(torch.clamp_min(sum_relu, 2.0**(-param.fl))).quant()
        sm = approx_div(relu_x, sum_relu)
        sm_floored = FixedTensor(torch.clamp_min(sm, 2.0**(-param.fl)))
        res = log(sm_floored, math.e)
        ctx.save_for_backward(sm, Tensor([dim]))
        return res

    backward = staticmethod(_logsoftmax_backward)


class PlaintextSoftMax(torch.autograd.Function):
    """Plaintext-exact softmax for ablation: torch.softmax then quant the output.
       No protocol-level approximation; only the trailing fixed-point truncation
       remains. Used to isolate the trunc protocol's impact from softmax error.
    """
    @staticmethod
    def forward(ctx, input, dim=-1):
        res = FixedTensor(torch.softmax(input, dim=dim)).quant()
        ctx.save_for_backward(res, Tensor([dim]))
        return res

    backward = staticmethod(_softmax_backward)


class PlaintextLogSoftMax(torch.autograd.Function):
    """Plaintext-exact log_softmax. Backward stores plaintext softmax (consistent
       with all other LogSoftMax variants saving their own protocol's softmax).
    """
    @staticmethod
    def forward(ctx, input, dim=-1):
        res = FixedTensor(torch.log_softmax(input, dim=dim)).quant()
        sm_value = FixedTensor(torch.softmax(input, dim=dim)).quant()
        ctx.save_for_backward(sm_value, Tensor([dim]))
        return res

    backward = staticmethod(_logsoftmax_backward)


class SigmaLogSoftMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim=-1):
        input_max = torch.max(input, dim=dim, keepdim=True).values
        intermdiate = input - input_max
        intermdiate_exp = sigma_exp(intermdiate)
        sum_ex = FixedTensor(torch.sum(intermdiate_exp, dim=dim, keepdim=True)).quant()
        logsum = log(sum_ex, math.e)
        res = intermdiate - logsum
        recip = sigma_recip(sum_ex)
        ctx.save_for_backward(FixedTensor(intermdiate_exp * recip).quant(), Tensor([dim]))
        return res

    backward = staticmethod(_logsoftmax_backward)


# Backward-compatible aliases.
ApproxSoftMax = PiranhaSubMaxSoftMax
ApproxLogSoftMax = PiranhaSubMaxLogSoftMax


def _dispatch_softmax(self, dim):
    t = param.softmax_type
    if t == 0:
        return PiranhaSubMaxSoftMax.apply(self, dim)
    elif t == 1:
        return PiranhaReluSoftMax.apply(self, dim)
    elif t == 2:
        return SigmaSoftMax.apply(self, dim)
    elif t == 3:
        return PlaintextSoftMax.apply(self, dim)
    raise ValueError("unknown softmax_type={}".format(t))


def _dispatch_log_softmax(self, dim):
    t = param.softmax_type
    if t == 0:
        return PiranhaSubMaxLogSoftMax.apply(self, dim)
    elif t == 1:
        return PiranhaReluLogSoftMax.apply(self, dim)
    elif t == 2:
        return SigmaLogSoftMax.apply(self, dim)
    elif t == 3:
        return PlaintextLogSoftMax.apply(self, dim)
    raise ValueError("unknown softmax_type={}".format(t))



class FixedTensor(torch.Tensor): 

    def round(self):
        if self.dtype == torch.bool:
            return FixedTensor(self)
        self = FixedTensor(torch.div(torch.round(torch.mul(self,2**param.fl)), 2**param.fl))
        return self
    
    def quant(self, wl=None, bitlength=None, fl=None, trunc_type=None):
        if self.dtype == torch.bool:
            return FixedTensor(self)
        # Honor caller overrides; fall back to current Parameter values.
        # Per-call override is needed by SIGMA-style LUT simulation, where
        # LUT input/output may use different fl than the global setting.
        wl = param.wl if wl is None else wl
        bitlength = param.bitlength if bitlength is None else bitlength
        fl = param.fl if fl is None else fl
        trunc_type = param.trunc_type if trunc_type is None else trunc_type
        self = FixedTensor(fixed_point_quantize(self, wl=wl, bitlength=bitlength, fl=fl, trunc_type=trunc_type, clamp=True, rounding="prob_error", trunc=True))
        return self

    def __mul__(self, other):
        # result = super().__mul__(other)
        # result.data = FixedTensor(result.data).quant()
        return MulFunction.apply(self, other)

    def __div__(self, other):
        return DivFunction.apply(self, other)
 
 
    def div(self, other):
        return DivFunction.apply(self, other)   
    def exp(self):
        return ExpFunction.apply(self)
    
    def inversqrt(self):
        return InverSqrtFunction.apply(self)
    
    def sqrt(self):
        return SqrtFunction.apply(self)
    
    def mean(self, dim = None, keepdim=False):
        return MeanFunction.apply(self, dim, keepdim)
    
    def std(self, dim = None, keepdim=False,unbiased=False):
        return StdFunction.apply(self, dim, keepdim, unbiased)
    
    def softmax(self, dim=-1):
        return _dispatch_softmax(self, dim)

    def log_softmax(self, dim=-1):
        return _dispatch_log_softmax(self, dim)
    
    def log(self, b):
        return LogFunction.apply(self, b)
    

    
     
