import torch
import torch.nn.functional as F
from Parameter import Parameter
from FixedTensor import FixedTensor, approx_exp, approx_div
from torch import Tensor
__all__ = ['ApproxSigmoid', 'ReLU', 'ReLU6', 'Sigmoid', 'Hardsigmoid', 'Hardswish']

param = Parameter()

class ApproxSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, sigmoid_spline):
        result = sigmoid_spline.calculate(FixedTensor(input))
        result.data = result.data.quant()
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = FixedTensor(grad_output).quant()
        result, = ctx.saved_tensors
        quant_res = FixedTensor(result)
        grad_input = FixedTensor(grad_output * FixedTensor(quant_res * (1 - quant_res)).quant()).quant()
        return grad_input, None
    


class Sigmoid(torch.nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}


    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Sigmoid.png

    Examples::

        >>> m = nn.Sigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def __init__(self, fixed = True, wl = param.wl, fl = param.fl, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fixed = fixed
        self.wl = wl 
        self.fl = fl
    
    def forward(self, input: FixedTensor) -> FixedTensor:
        if self.fixed:
            return ApproxSigmoid.apply(input)
        else:
            return torch.nn.Sigmoid(input)
class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ge0 = torch.ge(input, 0)
        # res = FixedTensor(torch.maximum(input, torch.tensor(0))).quant()
        ctx.save_for_backward(input)
        return input.clamp(min=0)  
    
    @staticmethod
    def backward(ctx, grad_output):
        if not isinstance(grad_output, FixedTensor):
            grad_output = FixedTensor(grad_output).quant()
        input, = ctx.saved_tensors  # 从上下文中恢复前向传播时保存的输入
        grad_input = grad_output.clone()  # 复制上游梯度
        grad_input[input < 0] = 0  # 应用ReLU的导数
        return grad_input
            
class ReLU(torch.nn.Module):
    r"""Applies the rectified linear unit function element-wise:

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ReLU.png

    Examples::

        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)


      An implementation of CReLU - https://arxiv.org/abs/1603.05201

        >>> m = nn.ReLU()
        >>> input = torch.randn(2).unsqueeze(0)
        >>> output = torch.cat((m(input), m(-input)))
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return ReLUFunction.apply(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class HardSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        tmp = input + 3.0
        ge0 = torch.ge(tmp, 0)
        le6 = torch.le(tmp, 6)
        res = torch.mul(torch.mul(tmp, ge0), le6)
        res = FixedTensor(torch.div(res, 6.0)).quant()
        ctx.save_for_backward(ge0, le6)
        return res
    
    @staticmethod
    def backward(ctx, grad_output):
        if not isinstance(grad_output, FixedTensor):
            grad_output = FixedTensor(grad_output).quant()
        ge0, le6 = ctx.saved_tensors
        grad_input = FixedTensor(torch.div(grad_output, 6)).quant()* ge0 * le6
        return grad_input
    
class Hardsigmoid(torch.nn.Module):
    r"""Applies the rectified linear unit function element-wise:

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ReLU.png

    Examples::

        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)


      An implementation of CReLU - https://arxiv.org/abs/1603.05201

        >>> m = nn.ReLU()
        >>> input = torch.randn(2).unsqueeze(0)
        >>> output = torch.cat((m(input), m(-input)))
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return HardSigmoidFunction.apply(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
        
class HardSwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        tmp = input + 3.0
        ge0 = torch.ge(tmp, 0)
        le6 = torch.le(tmp, 6)
        res = torch.mul(torch.mul(tmp, ge0), le6)
        res = FixedTensor(torch.div(res, 6.0)).quant()
        ge6 = ~le6
        res = res * input + FixedTensor(torch.mul(input, ge6)).quant()
        ctx.save_for_backward(input, ge0, le6)
        return res
    
    @staticmethod
    def backward(ctx, grad_output):
        if not isinstance(grad_output, FixedTensor):
            grad_output = FixedTensor(grad_output).quant()
        input, ge0, le6 = ctx.saved_tensors
        tmp_grad = FixedTensor(torch.div(2*input +3, 6)).quant()
        ge6 = ~le6
        grad_input = FixedTensor(grad_output * tmp_grad).quant()* ge0 * le6
        grad_input += grad_output * ge6
        return grad_input


class Hardswish(torch.nn.Module):
    r"""Applies the Hardswish function, element-wise, as described in the paper:
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_.

    Hardswish is defined as:

    .. math::
        \text{Hardswish}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Hardswish.png

    Examples::

        >>> m = nn.Hardswish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    __constants__ = ['inplace']

    inplace: bool

    def __init__(self, inplace : bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return HardSwishFunction.apply(input)  
    
class Relu6Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ge0 = torch.ge(input, 0)
        le6 = torch.le(input, 6)
        res = torch.mul(torch.mul(input, ge0), le6)
        res = FixedTensor(res).quant()
        ge6 = ~le6
        res = res + FixedTensor(torch.mul(6, ge6)).quant()
        ctx.save_for_backward(ge0, le6)
        return res
    
    @staticmethod
    def backward(ctx, grad_output):
        if not isinstance(grad_output, FixedTensor):
            grad_output = FixedTensor(grad_output).quant()
        ge0, le6 = ctx.saved_tensors
        grad_input = FixedTensor(grad_output).quant()* ge0 * le6
        return grad_input


class ReLU6(torch.nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.ReLU6()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
        
    def forward(self, input: Tensor) -> Tensor:
        return Relu6Function.apply(input)  