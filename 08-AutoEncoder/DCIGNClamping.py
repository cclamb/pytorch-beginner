
import torch

class DCIGNClamping(torch.autograd.Function):

    latent_dim = 0

    @staticmethod
    def forward(ctx, data_in, index):
        mask = torch.ones([DCIGNClamping.latent_dim, ], dtype=torch.bool)
        mask[index] = False
        ctx.save_for_backward(data_in, index, mask)
        batch_index = 0
        data_in[:, mask] = torch.mean(data_in[:, mask], dim=batch_index, keepdim=True)
        return data_in

    @staticmethod
    def backward(ctx, grad_output):
        data_in, index, mask = ctx.saved_tensors
        grad_input = grad_output.clone()
        batch_index = 0
        grad_input[:, mask] = data_in[:, mask] - torch.mean(data_in[:, mask], dim=batch_index, keepdim=True)
        return grad_input, None
