import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class SparsemaxFunction(Function):
    @staticmethod
    def forward(ctx, input, dim=-1):
        original_shape = input.shape
        input_reshape = input.reshape(-1, input.shape[dim])
        
        # Sort input in descending order
        sorted_input, _ = torch.sort(input_reshape, dim=1, descending=True)
        
        # Calculate rho
        z_cumsum = torch.cumsum(sorted_input, dim=1)
        k = torch.arange(1, input_reshape.shape[1] + 1, device=input.device)
        is_gt = (1 + k * sorted_input) > z_cumsum
        
        # Find k_max
        k_max = torch.sum(is_gt, dim=1, keepdim=True)
        
        # Calculate tau
        tau = (torch.gather(z_cumsum, 1, k_max - 1) - 1) / k_max
        
        # Calculate output
        output = torch.clamp(input_reshape - tau, min=0.0)
        output = output.reshape(original_shape)
        
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        dim = ctx.dim
        
        nonzeros = torch.ne(output, 0)
        sum_grad = torch.sum(grad_output * nonzeros, dim=dim, keepdim=True)
        num_nonzeros = torch.sum(nonzeros, dim=dim, keepdim=True)
        
        grad_input = grad_output - (sum_grad / num_nonzeros)
        grad_input = grad_input * nonzeros
        return grad_input, None

def sparsemax(input, dim=-1):
    return SparsemaxFunction.apply(input, dim)


class GBN(nn.Module):
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)
        
    def forward(self, x):
        if self.virtual_batch_size >= x.size(0):
            return self.bn(x)
            
        chunks = int(torch.ceil(torch.tensor(x.size(0) / self.virtual_batch_size)))
        res = [self.bn(x_) for x_ in torch.chunk(x, chunks, dim=0)]
        return torch.cat(res, dim=0)


class GLUBlock(nn.Module):
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02):
        super(GLUBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2, bias=False)
        self.bn = GBN(output_dim * 2, virtual_batch_size=virtual_batch_size, momentum=momentum)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out_dim = x.shape[1] // 2
        return x[:, :out_dim] * torch.sigmoid(x[:, out_dim:])


class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, n_shared=2, n_independent=2, virtual_batch_size=128, momentum=0.02):
        super(FeatureTransformer, self).__init__()
        self.shared = nn.ModuleList([
            GLUBlock(input_dim if i == 0 else output_dim, output_dim, virtual_batch_size, momentum)
            for i in range(n_shared)
        ])
        self.independent = nn.ModuleList([
            GLUBlock(input_dim if (i == 0 and n_shared == 0) else output_dim, output_dim, virtual_batch_size, momentum)
            for i in range(n_independent)
        ])
        self.scale = torch.sqrt(torch.tensor(0.5))

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.shared):
            if i == 0:
                out = layer(out)
            else:
                out = (out + layer(out)) * self.scale
                
        for layer in self.independent:
            out = (out + layer(out)) * self.scale
            
        return out


class AttentiveTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02, mask_type='sparsemax'):
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.bn = GBN(output_dim, virtual_batch_size=virtual_batch_size, momentum=momentum)
        self.mask_type = mask_type

    def forward(self, x, prior_scales):
        x = self.fc(x)
        x = self.bn(x)
        x = x * prior_scales
        if self.mask_type == 'sparsemax':
            x = sparsemax(x, dim=-1)
        else:
            x = F.softmax(x, dim=-1)
        return x


class TabNetEncoder(nn.Module):
    def __init__(self, input_dim, n_d=16, n_a=16, n_steps=3, gamma=1.3,
                 n_shared=2, n_independent=2, virtual_batch_size=128, momentum=0.02, mask_type='sparsemax'):
        super(TabNetEncoder, self).__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.mask_type = mask_type

        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=momentum)

        self.initial_transformer = FeatureTransformer(
            input_dim, n_d + n_a, n_shared, n_independent, virtual_batch_size, momentum
        )
        
        self.attentive_transformers = nn.ModuleList([
            AttentiveTransformer(n_a, input_dim, virtual_batch_size, momentum, mask_type)
            for _ in range(n_steps)
        ])
        self.feature_transformers = nn.ModuleList([
            FeatureTransformer(input_dim, n_d + n_a, n_shared, n_independent, virtual_batch_size, momentum)
            for _ in range(n_steps)
        ])

    def forward(self, x):
        x = self.initial_bn(x)
        
        prior_scales = torch.ones(x.shape, device=x.device)
        
        out = self.initial_transformer(x)
        a = out[:, self.n_d:]
        
        steps_output = []
        masks = []
        
        for step in range(self.n_steps):
            mask = self.attentive_transformers[step](a, prior_scales)
            masks.append(mask)
            
            prior_scales = prior_scales * (self.gamma - mask)
            
            masked_x = x * mask
            out = self.feature_transformers[step](masked_x)
            
            d = out[:, :self.n_d]
            a = out[:, self.n_d:]
            
            steps_output.append(d)
            
        res = sum(steps_output)
        return res, masks


class TabNet(nn.Module):
    def __init__(self, input_dim: int, n_d: int = 16, n_a: int = 16, n_steps: int = 3, gamma: float = 1.3,
                 lambda_sparse: float = 0.001, momentum: float = 0.02, virtual_batch_size: int = 128, mask_type: str = 'sparsemax'):
        super(TabNet, self).__init__()
        self.lambda_sparse = lambda_sparse
        self.encoder = TabNetEncoder(
            input_dim=input_dim, n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
            virtual_batch_size=virtual_batch_size, momentum=momentum, mask_type=mask_type
        )
        self.fc = nn.Linear(n_d, 1)

    def forward(self, x):
        res, masks = self.encoder(x)
        out = torch.sigmoid(self.fc(res)).squeeze(dim=-1)
        
        loss = torch.tensor(0.0, device=x.device)
        for m in masks:
            loss -= torch.mean(torch.sum(m * torch.log(m + 1e-15), dim=-1))
            
        sparsity_loss = self.lambda_sparse * loss
        return out, sparsity_loss
