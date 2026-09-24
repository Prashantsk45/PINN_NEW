"""
Fourier-ResPINN Neural Network Architecture for 3D Hydrogen Swirl Combustor.
Includes:
- Gaussian Random Fourier Feature Layer (Tancik et al. NeurIPS 2020)
- 6-Block Deep Residual Backbone with Swish / SiLU Activation
- Three Decoupled Output Heads (Hydrodynamic, Thermal, Species)
- Strict Positivity Enforcement via Softplus (Species & Eddy Viscosity)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

class GaussianFourierFeatures(nn.Module):
    """
    Gaussian Random Fourier Features mapping:
    gamma(x) = [cos(2 * pi * B * x), sin(2 * pi * B * x), phi]^T
    Overcomes spectral bias of MLPs to resolve sub-millimeter flame fronts.
    """
    def __init__(self, in_features=3, num_fourier_features=config.FOURIER_NUM_FEATURES, sigma=config.FOURIER_SCALE_SIGMA):
        super().__init__()
        self.in_features = in_features
        self.num_fourier_features = num_fourier_features
        self.sigma = sigma
        
        # Fixed random Gaussian projection matrix B (non-trainable)
        B = torch.randn(num_fourier_features, in_features) * sigma
        self.register_buffer("B", B)

    def forward(self, x_spatial, phi):
        """
        Args:
            x_spatial: (N, 3) tensor [x_tilde, y_tilde, z_tilde]
            phi: (N, 1) tensor [phi_tilde]
        Returns:
            (N, 2 * num_fourier_features + 1) tensor
        """
        # (N, 3) @ (3, num_features) -> (N, num_features)
        x_proj = 2.0 * math.pi * torch.matmul(x_spatial, self.B.t())
        cos_features = torch.cos(x_proj)
        sin_features = torch.sin(x_proj)
        
        # Concatenate Fourier features with operational equivalence ratio phi
        out = torch.cat([cos_features, sin_features, phi], dim=-1)
        return out

class ResNetBlock(nn.Module):
    """
    Two-layer residual block with skip connection and SiLU activation.
    a^(l+1) = SiLU(W2 * SiLU(W1 * a^(l) + b1) + b2) + a^(l)
    """
    def __init__(self, hidden_dim=config.RESNET_HIDDEN_DIM):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

        # Xavier initialization
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        residual = x
        out = self.act(self.linear1(x))
        out = self.act(self.linear2(out))
        return out + residual

class FourierResPINN(nn.Module):
    """
    Multi-Head Physics-Informed Neural Network with Fourier Embeddings.
    Inputs: [x_tilde, y_tilde, z_tilde, phi_tilde] (4 dimensions)
    Outputs:
      - Hydrodynamic Head: [u_tilde, v_tilde, w_tilde, p_tilde, nu_t_tilde] (5 outputs)
      - Thermal Head: [T_tilde] (1 output)
      - Species Head: [Y_H2, Y_O2, Y_H2O, Y_OH] (4 outputs)
    """
    def __init__(self, 
                 fourier_sigma=config.FOURIER_SCALE_SIGMA,
                 num_fourier=config.FOURIER_NUM_FEATURES,
                 num_blocks=config.RESNET_NUM_BLOCKS,
                 hidden_dim=config.RESNET_HIDDEN_DIM):
        super().__init__()
        
        # 1. Fourier Feature Layer
        self.fourier_layer = GaussianFourierFeatures(in_features=3, num_fourier_features=num_fourier, sigma=fourier_sigma)
        fourier_out_dim = 2 * num_fourier + 1  # 64*2 + 1 = 129
        
        # 2. Input Projection Layer
        self.input_layer = nn.Linear(fourier_out_dim, hidden_dim)
        self.act = nn.SiLU()
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        
        # 3. ResNet Backbone
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim) for _ in range(num_blocks)])
        
        # 4. Decoupled Output Heads
        # Head 1: Hydrodynamic [u, v, w, p, nu_t]
        self.hydro_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 5)
        )
        
        # Head 2: Thermal & Wall Heat Flux [T_tilde, q_wall_tilde]
        self.thermal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Head 3: Species & Emissions [Y_H2, Y_O2, Y_H2O, X_OH, X_NO]
        self.species_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 5)
        )

    def forward(self, x_in):
        """
        Args:
            x_in: (N, 4) tensor containing [x_tilde, y_tilde, z_tilde, phi_tilde]
        Returns:
            dict containing:
                'u': (N, 1) non-dim x-velocity
                'v': (N, 1) non-dim y-velocity
                'w': (N, 1) non-dim z-axial velocity
                'p': (N, 1) non-dim static pressure
                'nu_t': (N, 1) non-dim turbulent eddy viscosity (Softplus >= 0)
                'T': (N, 1) non-dim static temperature
                'q_wall': (N, 1) non-dim wall heat flux
                'Y_H2': (N, 1) mass fraction H2 (Softplus >= 0)
                'Y_O2': (N, 1) mass fraction O2 (Softplus >= 0)
                'Y_H2O': (N, 1) mass fraction H2O (Softplus >= 0)
                'X_OH': (N, 1) mole fraction OH radical (Softplus >= 0)
                'Y_OH': alias to X_OH
                'X_NO': (N, 1) mole fraction NO pollutant (Softplus >= 0)
        """
        x_spatial = x_in[:, :3]
        phi = x_in[:, 3:4]
        
        # 1. Fourier feature embedding
        features = self.fourier_layer(x_spatial, phi)
        h = self.act(self.input_layer(features))
        
        # 2. ResNet propagation
        for block in self.blocks:
            h = block(h)
            
        # 3. Decoupled Head Evaluation
        hydro_raw = self.hydro_head(h)
        thermal_raw = self.thermal_head(h)
        species_raw = self.species_head(h)
        
        # 4. Physical Transformations & Bounds
        u = hydro_raw[:, 0:1]
        v = hydro_raw[:, 1:2]
        w = hydro_raw[:, 2:3]
        p = hydro_raw[:, 3:4]
        # Softplus guarantees non-negative turbulent eddy viscosity
        nu_t = F.softplus(hydro_raw[:, 4:5])
        
        # Temperature output (normalized in [0, 1.2]) and Wall Heat Flux
        T = thermal_raw[:, 0:1]
        q_wall = thermal_raw[:, 1:2]
        
        # Softplus guarantees non-negative species mass fractions and emissions
        Y_H2 = F.softplus(species_raw[:, 0:1])
        Y_O2 = F.softplus(species_raw[:, 1:2])
        Y_H2O = F.softplus(species_raw[:, 2:3])
        X_OH = F.softplus(species_raw[:, 3:4])
        X_NO = F.softplus(species_raw[:, 4:5])
        
        return {
            'u': u,
            'v': v,
            'w': w,
            'p': p,
            'nu_t': nu_t,
            'T': T,
            'q_wall': q_wall,
            'Y_H2': Y_H2,
            'Y_O2': Y_O2,
            'Y_H2O': Y_H2O,
            'X_OH': X_OH,
            'Y_OH': X_OH,
            'X_NO': X_NO
        }

if __name__ == "__main__":
    net = FourierResPINN()
    num_params = sum(p.numel() for p in net.parameters())
    print("--- Model Architecture Verification ---")
    print(f"Total Trainable Parameters: {num_params:,} (~{num_params*4/1024:.1f} KB)")
    
    # Test forward pass with dummy tensor
    x_dummy = torch.randn(16, 4, requires_grad=True)
    out = net(x_dummy)
    print("Forward pass successful! Output fields:")
    for k, v in out.items():
        print(f"  {k:<8}: Shape {tuple(v.shape)}, Min={v.min().item():.3f}, Max={v.max().item():.3f}")
