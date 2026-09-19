import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List

class KANLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3, scale_noise: float = 0.1, scale_base: float = 1.0, scale_spline: float = 1.0, base_activation=torch.nn.SiLU, grid_eps: float = 0.02, grid_range: List[float] = [-1.0, 1.0]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Grid definition
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        
        self.base_activation = base_activation()
        
        # Parameters
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        # Initialize
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=scale_noise / grid_size)

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline basis functions.
        x: (batch, in_features)
        Returns: (batch, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        grid = self.grid # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1) # (batch, in_features, 1)
        
        # Initialize 0-th order B-splines
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        for k in range(1, self.spline_order + 1):
            left_term = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)])
            right_term = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-(k)])
            
            # Avoid division by zero by setting NaNs to 0
            left_term = torch.nan_to_num(left_term)
            right_term = torch.nan_to_num(right_term)
            
            bases = left_term * bases[:, :, :-1] + right_term * bases[:, :, 1:]
            
        return bases.contiguous()

    def forward(self, x: torch.Tensor):
        """
        x: (batch, in_features)
        """
        # Base computation
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # Spline computation
        splines = self.b_splines(x) # (batch, in_features, grid_size + spline_order)
        
        # spline_weight: (out_features, in_features, grid_size + spline_order)
        # We want to multiply and sum over in_features and the basis dimension
        spline_output = torch.einsum('oik,bik->bo', self.spline_weight, splines)
        
        return base_output + spline_output


class KAN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int = 1, grid_size: int = 5, spline_order: int = 3, dropout: float = 0.1):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        layer_dims = [input_dim] + hidden_layers + [output_dim]
        
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                KANLinear(
                    in_features=layer_dims[i],
                    out_features=layer_dims[i+1],
                    grid_size=grid_size,
                    spline_order=spline_order
                )
            )
            if i < len(layer_dims) - 2: # No norm/dropout after the last layer
                self.layer_norms.append(nn.LayerNorm(layer_dims[i+1]))
                self.dropouts.append(nn.Dropout(dropout))
                
    def forward(self, x: torch.Tensor):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.layers) - 1:
                x = self.layer_norms[i](x)
                x = self.dropouts[i](x)
        
        # Binary classification -> Sigmoid
        return torch.sigmoid(x).squeeze(dim=-1)
