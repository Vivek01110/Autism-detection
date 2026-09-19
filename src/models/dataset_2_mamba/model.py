"""
Mamba (Selective State Space Model) for Dataset 2 Eye-Tracking Sequence Classification.

Implementation Details:
-----------------------
- Architecture: Pure PyTorch implementation of the Mamba / S6 Selective State Space Model
  (Gu & Dao, 2023: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces").
- External packages: None required; runs natively on CPU, MPS, and CUDA without custom C++/Triton binaries.
- Input shape: (B, 200, 8) where B is batch size, 200 is sequence length, and 8 is eye-tracking channels.
- Number of layers: Configurable (default 2 residual Mamba blocks).
- Model dimension (d_model): Configurable (default 64).
- Inner dimension (d_inner): expand * d_model (default 2 * 64 = 128).
- State dimension (d_state): Configurable (default 16).
- Convolution width (d_conv): Configurable 1D causal depthwise convolution (default 4).
- Discretization: Zero-Order Hold (ZOH) with input-dependent step size Delta = softplus(Linear(u) + Delta_bias).
- State matrix A: Parameterized via HiPPO structured matrix log-initialization (A_dn = -(n + 1)).
- Gating branch: Multiplicative SiLU gating with parallel projection z.
- Dropout: Configurable (default 0.1).
- Pooling strategy: Temporal mean pooling across the 200 timesteps (B, 200, d_model) -> (B, d_model).
- Classification head: LayerNorm(d_model) -> Linear(d_model, 32) -> GELU() -> Dropout(0.1) -> Linear(32, 1) -> Sigmoid().
- Output: ASD probability in [0, 1].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SelectiveSSM(nn.Module):
    """
    Continuous-to-discrete Selective State Space Model (S6).
    Parameterizes Delta, B, and C as input-dependent projections of u.
    """

    def __init__(self, d_inner: int = 128, d_state: int = 16, dt_rank: Optional[int] = None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(16, d_inner // 16)

        # Input projection for Delta, B, and C
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # Initialize Delta projection bias to encourage initial Delta ~ 0.001 - 0.1
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Initialize HiPPO structured state matrix A: shape (d_inner, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # Learnable skip parameter D: shape (d_inner,)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Input tensor of shape (B, L, d_inner)
        Returns:
            y: SSM output tensor of shape (B, L, d_inner)
        """
        B, L, D = u.shape
        N = self.d_state

        # Project input to selective parameters: Delta_proj, B_mat, C_mat
        x_dbl = self.x_proj(u)  # (B, L, dt_rank + 2*d_state)
        delta_rank, B_mat, C_mat = torch.split(x_dbl, [self.dt_rank, N, N], dim=-1)

        # Compute continuous-to-discrete step size Delta: (B, L, D)
        delta = F.softplus(self.dt_proj(delta_rank))

        # Discretize continuous state matrix A: A = -exp(A_log) <= 0
        A = -torch.exp(self.A_log)  # (D, N)
        # deltaA: (B, L, D, N)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # deltaB_u: (B, L, D, N)
        deltaB_u = (delta.unsqueeze(-1) * B_mat.unsqueeze(2)) * u.unsqueeze(-1)

        # Vectorized slice unpacking for fast scan iteration (avoids per-step slice creation)
        dA_list = deltaA.unbind(dim=1)
        dBu_list = deltaB_u.unbind(dim=1)
        C_list = C_mat.unbind(dim=1)

        # Sequential scan over sequence timesteps
        h = torch.zeros(B, D, N, device=u.device, dtype=u.dtype)
        ys = []
        for a_t, b_t, c_t in zip(dA_list, dBu_list, C_list):
            h = a_t * h + b_t
            y_t = torch.sum(h * c_t.unsqueeze(1), dim=-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1) + u * self.D
        return y


class MambaBlock(nn.Module):
    """
    Mamba Residual Block combining LayerNorm, In-Projection, 1D Causal Depthwise Conv,
    Selective SSM, Multiplicative Gating, and Out-Projection.
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_inner = d_model * expand
        self.d_model = d_model
        self.d_inner = d_inner

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )
        self.ssm = SelectiveSSM(d_inner=d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, L, d_model)
        Returns:
            Tensor of shape (B, L, d_model)
        """
        residual = x
        x_norm = self.norm(x)

        # Project to 2 * d_inner and split into u and gating branch z
        u_z = self.in_proj(x_norm)
        u, z = torch.chunk(u_z, 2, dim=-1)

        # 1D Causal Depthwise Convolution: transpose to (B, d_inner, L)
        u_conv = self.conv1d(u.transpose(1, 2))[:, :, : x.shape[1]].transpose(1, 2)
        u_conv = F.silu(u_conv)

        # Selective State Space Model
        y_ssm = self.ssm(u_conv)

        # Multiplicative gating with branch z
        y_gated = y_ssm * F.silu(z)

        # Output projection and residual connection
        out = self.out_proj(y_gated)
        out = self.dropout(out)
        return residual + out


class MambaClassifier(nn.Module):
    """
    Full Mamba Architecture for Eye-Tracking Binary ASD Classification.
    """

    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_layers = n_layers

        # Input feature projection (8 channels -> d_model)
        self.input_proj = nn.Linear(input_dim, d_model)

        # Stack of Mamba residual blocks
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        # Classification head with temporal mean pooling
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Eye-tracking sequence tensor of shape (B, 200, input_dim)
        Returns:
            ASD probability tensor of shape (B,) with values in [0, 1]
        """
        # Feature projection: (B, L, input_dim) -> (B, L, d_model)
        h = self.input_proj(x)

        # Pass through sequential Mamba blocks
        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)

        # Temporal mean pooling across the sequence length (L=200)
        h_pool = torch.mean(h, dim=1)  # (B, d_model)

        # Classification head -> Sigmoid probability
        prob = self.head(h_pool).squeeze(-1)
        return prob
