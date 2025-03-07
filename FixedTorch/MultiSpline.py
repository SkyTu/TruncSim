import torch


class MultiSpline:
    def __init__(self, spline_length ,spline_list, function_list):
        print(spline_length, len(spline_list), len(function_list))
        assert(spline_length == len(spline_list) + 1 == (len(function_list)))
        self.length = spline_length
        self.spline = spline_list
        self.function = function_list
    
    def calculate(self, tensor):
        select = torch.le(tensor, self.spline[0])
        result = select * self.function[0](tensor)
        tensor.requires_grad = True
        for idx in range(1, self.length - 1):
            select = torch.gt(tensor, self.spline[idx-1]) * torch.le(tensor, self.spline[idx])
            result += select * self.function[idx](tensor)
        select = torch.gt(tensor, self.spline[self.length - 2])
        result += select * self.function[self.length - 1](tensor)
        return result