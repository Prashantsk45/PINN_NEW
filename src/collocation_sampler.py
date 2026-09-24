"""
Meshless Collocation Point Sampler for 3D Swirl Combustor PINN.
Generates interior PDE collocation points via Latin Hypercube Sampling (LHS)
and boundary condition collocation points (Inlet, Dump Step, Liner Wall, Outlet).
"""

import numpy as np
import torch
from scipy.stats import qmc

import config

class CombustorCollocationSampler:
    """
    Generates space-filling meshless collocation points in the cylindrical
    combustor domain: R <= 42.5 mm, L <= 110 mm, Phi in [0.55, 1.00].
    """
    def __init__(self, seed=config.RANDOM_SEED):
        self.seed = seed
        self.sampler = qmc.LatinHypercube(d=4, seed=seed)

    def sample_interior_points(self, n_points=config.N_PDE_COLLOCATION):
        """
        Samples n_points uniformly in the cylindrical volume:
        r = R_chamber * sqrt(xi_1)
        theta = 2 * pi * xi_2
        x = r * cos(theta), y = r * sin(theta)
        z = L_chamber * xi_3
        phi in [0.55, 1.00]
        
        Returns:
            torch.Tensor of shape (n_points, 4): [x_tilde, y_tilde, z_tilde, phi_tilde]
        """
        lhs_samples = self.sampler.random(n=n_points)
        
        xi_r = lhs_samples[:, 0]
        xi_theta = lhs_samples[:, 1]
        xi_z = lhs_samples[:, 2]
        xi_phi = lhs_samples[:, 3]
        
        # Cylindrical to Cartesian coordinates [m]
        r = config.R_CHAMBER * np.sqrt(xi_r)
        theta = 2.0 * np.pi * xi_theta
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = config.L_CHAMBER * xi_z
        phi = 0.55 + (1.00 - 0.55) * xi_phi
        
        # Non-dimensionalize
        x_tilde = x / config.L_0
        y_tilde = y / config.L_0
        z_tilde = z / config.L_0
        phi_tilde = phi
        
        X_pde = np.stack([x_tilde, y_tilde, z_tilde, phi_tilde], axis=1)
        return torch.tensor(X_pde, dtype=config.TORCH_DTYPE, requires_grad=True)

    def sample_boundary_points(self, n_points_per_bc=2500):
        """
        Generates boundary collocation points across:
        1. Inlet (Swirler face): z = 0, R_hub <= r <= R_tip
        2. Dump Wall: z = 0, R_tip < r <= R_chamber
        3. Liner Wall: r = R_chamber, 0 <= z <= L_chamber
        4. Outlet: z = L_chamber, 0 <= r <= R_chamber
        """
        rng = np.random.default_rng(self.seed)
        
        # 1. Swirler Inlet (z = 0, annular swirler)
        r_in = np.sqrt(rng.uniform(config.R_HUB**2, config.R_TIP**2, n_points_per_bc))
        th_in = rng.uniform(0, 2*np.pi, n_points_per_bc)
        x_in = r_in * np.cos(th_in)
        y_in = r_in * np.sin(th_in)
        z_in = np.zeros(n_points_per_bc)
        phi_in = rng.uniform(0.55, 1.00, n_points_per_bc)
        
        inlet_coords = np.stack([x_in/config.L_0, y_in/config.L_0, z_in/config.L_0, phi_in], axis=1)
        
        # 2. Dump Step Wall (z = 0, outer annular face)
        r_dump = np.sqrt(rng.uniform(config.R_TIP**2, config.R_CHAMBER**2, n_points_per_bc))
        th_dump = rng.uniform(0, 2*np.pi, n_points_per_bc)
        x_dump = r_dump * np.cos(th_dump)
        y_dump = r_dump * np.sin(th_dump)
        z_dump = np.zeros(n_points_per_bc)
        phi_dump = rng.uniform(0.55, 1.00, n_points_per_bc)
        
        dump_coords = np.stack([x_dump/config.L_0, y_dump/config.L_0, z_dump/config.L_0, phi_dump], axis=1)
        
        # 3. Combustor Outer Liner Wall (r = R_chamber)
        th_wall = rng.uniform(0, 2*np.pi, n_points_per_bc)
        x_wall = config.R_CHAMBER * np.cos(th_wall)
        y_wall = config.R_CHAMBER * np.sin(th_wall)
        z_wall = rng.uniform(0, config.L_CHAMBER, n_points_per_bc)
        phi_wall = rng.uniform(0.55, 1.00, n_points_per_bc)
        
        wall_coords = np.stack([x_wall/config.L_0, y_wall/config.L_0, z_wall/config.L_0, phi_wall], axis=1)
        
        # 4. Combustor Outlet (z = L_chamber)
        r_out = config.R_CHAMBER * np.sqrt(rng.uniform(0, 1, n_points_per_bc))
        th_out = rng.uniform(0, 2*np.pi, n_points_per_bc)
        x_out = r_out * np.cos(th_out)
        y_out = r_out * np.sin(th_out)
        z_out = np.full(n_points_per_bc, config.L_CHAMBER)
        phi_out = rng.uniform(0.55, 1.00, n_points_per_bc)
        
        outlet_coords = np.stack([x_out/config.L_0, y_out/config.L_0, z_out/config.L_0, phi_out], axis=1)
        
        return {
            'inlet': torch.tensor(inlet_coords, dtype=config.TORCH_DTYPE, requires_grad=True),
            'dump': torch.tensor(dump_coords, dtype=config.TORCH_DTYPE, requires_grad=True),
            'wall': torch.tensor(wall_coords, dtype=config.TORCH_DTYPE, requires_grad=True),
            'outlet': torch.tensor(outlet_coords, dtype=config.TORCH_DTYPE, requires_grad=True)
        }

if __name__ == "__main__":
    sampler = CombustorCollocationSampler()
    pde_pts = sampler.sample_interior_points(10000)
    bc_dict = sampler.sample_boundary_points(1000)
    
    print("--- Collocation Sampler Verification ---")
    print(f"Interior PDE Points Shape: {pde_pts.shape}")
    print(f"Inlet BC Points Shape:     {bc_dict['inlet'].shape}")
    print(f"Dump Wall Points Shape:    {bc_dict['dump'].shape}")
    print(f"Liner Wall Points Shape:   {bc_dict['wall'].shape}")
    print(f"Outlet BC Points Shape:    {bc_dict['outlet'].shape}")
    print("Collocation point generator operating correctly!")
