from FixedTensor import FixedTensor
import torch
from torch.autograd import Function
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
import math
from typing import Union, Tuple, Any
__all__ = ['Linear', 'Identity']
class Identity(torch.nn.Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, input: FixedTensor) -> FixedTensor:
        return input
    
class Exp(Function):
    @staticmethod
    def forward(ctx, input):
        output = input *input
        return FixedTensor(output).quant()
    
class FixedPointLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        if bias is not None:
            ctx.save_for_backward(input, FixedTensor(weight), FixedTensor(bias))
        else:
            ctx.save_for_backward(input, FixedTensor(weight), None)
        output = FixedTensor(input.mm(weight.t())).quant()
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return FixedTensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_output = FixedTensor(grad_output)

        if ctx.needs_input_grad[0]:
            grad_input = FixedTensor(grad_output.mm(weight)).quant()
        if ctx.needs_input_grad[1]:
            grad_weight = FixedTensor(grad_output.t().mm(input)).quant()
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = FixedTensor(grad_output.sum(0)).quant()
        # print(grad_input, grad_weight, grad_bias)
        return grad_input, grad_weight, grad_bias

class Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: FixedTensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, fixed = True,
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # self.weight.data = FixedTensor(self.weight.data).quant()
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            # self.bias.data = FixedTensor(self.bias.data).quant()
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.weight)
        bound = math.sqrt(6.0 / (self.weight.shape[0]+self.weight.shape[1])) if fan_in > 0 else 0
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # bound = FixedTensor([bound]).round()
        # init.uniform_(self.weight, -bound[0], bound[0])
        # init.constant_(self.weight, 0.0)
        if self.bias is not None:
            init.constant_(self.bias, 0)
            self.bias.data = FixedTensor(self.bias.data).round()
        self.weight.data = FixedTensor(self.weight.data).round()

    def forward(self, input: FixedTensor) -> FixedTensor:
        return FixedPointLinearFunction.apply(input, self.weight, self.bias)
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


