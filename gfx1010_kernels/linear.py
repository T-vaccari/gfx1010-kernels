import torch
from torch.nn import functional as F


@torch.compiler.disable
def _weight_gradient(grad_output, input):
    return grad_output.t().contiguous() @ input


class _AutocastLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        input_half = input.to(torch.float16)
        weight_half = weight.to(torch.float16)
        bias_half = bias.to(torch.float16) if bias is not None else None
        output = F.linear(input_half, weight_half, bias_half)
        ctx.save_for_backward(input_half, weight_half)
        ctx.input_shape = input.shape
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_half, weight_half = ctx.saved_tensors
        grad_output_half = grad_output.to(torch.float16)
        grad_output_2d = grad_output_half.reshape(-1, grad_output_half.shape[-1])
        input_2d = input_half.reshape(-1, input_half.shape[-1])
        grad_input = (grad_output_2d @ weight_half).reshape(ctx.input_shape)
        grad_weight = _weight_gradient(grad_output_2d, input_2d)
        grad_bias = grad_output_2d.sum(0) if ctx.has_bias else None
        return grad_input, grad_weight.float(), grad_bias.float() if grad_bias is not None else None


def autocast_linear(input, weight, bias=None):
    if (
        input.is_cuda
        and input.dtype in (torch.float16, torch.float32)
        and weight.dtype == torch.float32
    ):
        return _AutocastLinear.apply(input, weight, bias)
    return F.linear(input, weight, bias)


class AutocastLinear(torch.nn.Linear):
    def forward(self, input):
        return autocast_linear(input, self.weight, self.bias)
