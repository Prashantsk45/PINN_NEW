"""
Physics Loss Engine for 3D Hydrogen Swirl Combustor DA-PINN.
Computes autograd derivatives for:
1. Continuity: Del . (rho * u) = 0
2. 3D Momentum: Favre Navier-Stokes with turbulent eddy viscosity
3. Radial Centrifugal Swirl Balance: dp/dr = rho * V_theta^2 / r
4. Sensible Enthalpy / Energy with chemical heat release
5. Multi-species transport with Arrhenius hydrogen kinetics
6. Boundary conditions (Inlet, Dump step, Liner wall, Exit)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

def compute_gradient(y, x):
    """
    Computes dy/dx using automatic differentiation with graph creation.
    """
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    if grad is None:
        return torch.zeros_like(x)
    return grad

class CombustorPhysicsLoss(nn.Module):
    """
    Evaluates multi-physics PDE residuals on collocation coordinates.
    """
    def __init__(self):
        super().__init__()
        self.re = config.RE
        self.pr = config.PR
        self.sc = config.SC
        self.da = config.DA
        self.pr_t = config.PR_T
        self.sc_t = config.SC_T
        self.t_ratio = (config.T_AD - config.T_0) / config.T_0  # ~ 6.93

    def compute_density(self, T_tilde):
        """
        Non-dimensional ideal gas density: rho_tilde = 1 / (1 + beta * T_tilde)
        """
        return 1.0 / (1.0 + self.t_ratio * torch.clamp(T_tilde, min=0.0, max=1.2))

    def compute_reaction_rate(self, T_tilde, Y_H2, Y_O2):
        """
        Dimensionless Arrhenius reaction rate for 2 H2 + O2 -> 2 H2O:
        omega_dot = Da * Y_H2^2 * Y_O2 * exp(-Ta / T)
        """
        T_dim = config.T_0 + (config.T_AD - config.T_0) * torch.clamp(T_tilde, min=0.05, max=1.2)
        T_a = 8000.0
        arrhenius = torch.exp(-T_a / T_dim)
        rate = self.da * (Y_H2 ** 2) * Y_O2 * arrhenius * 1e4
        return rate

    def pde_residuals(self, model, x_pde):
        """
        Evaluates the 5 governing physical conservation residuals on collocation points x_pde.
        Args:
            model: FourierResPINN model
            x_pde: (N, 4) tensor [x_tilde, y_tilde, z_tilde, phi_tilde] with requires_grad=True
        Returns:
            dict of residual losses:
                'loss_cont', 'loss_mom', 'loss_swirl', 'loss_energy', 'loss_species'
        """
        outputs = model(x_pde)
        
        u = outputs['u']
        v = outputs['v']
        w = outputs['w']
        p = outputs['p']
        nu_t = outputs['nu_t']
        T = outputs['T']
        Y_H2 = outputs['Y_H2']
        Y_O2 = outputs['Y_O2']
        Y_H2O = outputs['Y_H2O']
        Y_OH = outputs['Y_OH']
        
        x_coord = x_pde[:, 0:1]
        y_coord = x_pde[:, 1:2]
        
        # Density field
        rho = self.compute_density(T)
        
        # 1. First derivatives of velocity
        grad_u = compute_gradient(u, x_pde)
        grad_v = compute_gradient(v, x_pde)
        grad_w = compute_gradient(w, x_pde)
        
        du_dx = grad_u[:, 0:1]
        du_dy = grad_u[:, 1:2]
        du_dz = grad_u[:, 2:3]
        
        dv_dx = grad_v[:, 0:1]
        dv_dy = grad_v[:, 1:2]
        dv_dz = grad_v[:, 2:3]
        
        dw_dx = grad_w[:, 0:1]
        dw_dy = grad_w[:, 1:2]
        dw_dz = grad_w[:, 2:3]
        
        # Continuity Residual: Del . (rho * u) = rho (div u) + u . (grad rho)
        grad_rho = compute_gradient(rho, x_pde)
        div_u = du_dx + dv_dy + dw_dz
        res_cont = rho * div_u + (u * grad_rho[:, 0:1] + v * grad_rho[:, 1:2] + w * grad_rho[:, 2:3])
        
        # 2. Laplacians of velocity
        d2u_dx2 = compute_gradient(du_dx, x_pde)[:, 0:1]
        d2u_dy2 = compute_gradient(du_dy, x_pde)[:, 1:2]
        d2u_dz2 = compute_gradient(du_dz, x_pde)[:, 2:3]
        lap_u = d2u_dx2 + d2u_dy2 + d2u_dz2
        
        d2v_dx2 = compute_gradient(dv_dx, x_pde)[:, 0:1]
        d2v_dy2 = compute_gradient(dv_dy, x_pde)[:, 1:2]
        d2v_dz2 = compute_gradient(dv_dz, x_pde)[:, 2:3]
        lap_v = d2v_dx2 + d2v_dy2 + d2v_dz2
        
        d2w_dx2 = compute_gradient(dw_dx, x_pde)[:, 0:1]
        d2w_dy2 = compute_gradient(dw_dy, x_pde)[:, 1:2]
        d2w_dz2 = compute_gradient(dw_dz, x_pde)[:, 2:3]
        lap_w = d2w_dx2 + d2w_dy2 + d2w_dz2
        
        # Pressure gradients
        grad_p = compute_gradient(p, x_pde)
        dp_dx = grad_p[:, 0:1]
        dp_dy = grad_p[:, 1:2]
        dp_dz = grad_p[:, 2:3]
        
        diff_mom = (1.0 + nu_t) / (self.re / 1000.0)
        
        # Convection terms
        conv_u = u * du_dx + v * du_dy + w * du_dz
        conv_v = u * dv_dx + v * dv_dy + w * dv_dz
        conv_w = u * dw_dx + v * dw_dy + w * dw_dz
        
        res_mom_x = rho * conv_u + dp_dx - diff_mom * lap_u
        res_mom_y = rho * conv_v + dp_dy - diff_mom * lap_v
        res_mom_z = rho * conv_w + dp_dz - diff_mom * lap_w
        
        # 3. Radial Swirl Centrifugal Equilibrium: dp/dr = rho * V_theta^2 / r
        eps_r = 1e-4
        r_coord = torch.sqrt(x_coord**2 + y_coord**2 + eps_r**2)
        v_theta = (x_coord * v - y_coord * u) / r_coord
        dp_dr = (x_coord / r_coord) * dp_dx + (y_coord / r_coord) * dp_dy
        centrifugal_force = rho * (v_theta ** 2) / r_coord
        res_swirl = dp_dr - centrifugal_force
        
        # 4. Energy Residual
        grad_T = compute_gradient(T, x_pde)
        dT_dx = grad_T[:, 0:1]
        dT_dy = grad_T[:, 1:2]
        dT_dz = grad_T[:, 2:3]
        
        d2T_dx2 = compute_gradient(dT_dx, x_pde)[:, 0:1]
        d2T_dy2 = compute_gradient(dT_dy, x_pde)[:, 1:2]
        d2T_dz2 = compute_gradient(dT_dz, x_pde)[:, 2:3]
        lap_T = d2T_dx2 + d2T_dy2 + d2T_dz2
        
        conv_T = u * dT_dx + v * dT_dy + w * dT_dz
        diff_energy = (1.0 + nu_t / self.pr_t) / (self.re * self.pr / 1000.0)
        q_chem = self.compute_reaction_rate(T, Y_H2, Y_O2)
        res_energy = rho * conv_T - diff_energy * lap_T - q_chem
        
        # 5. Species Residual (H2 consumption)
        grad_H2 = compute_gradient(Y_H2, x_pde)
        dH2_dx = grad_H2[:, 0:1]
        dH2_dy = grad_H2[:, 1:2]
        dH2_dz = grad_H2[:, 2:3]
        
        d2H2_dx2 = compute_gradient(dH2_dx, x_pde)[:, 0:1]
        d2H2_dy2 = compute_gradient(dH2_dy, x_pde)[:, 1:2]
        d2H2_dz2 = compute_gradient(dH2_dz, x_pde)[:, 2:3]
        lap_H2 = d2H2_dx2 + d2H2_dy2 + d2H2_dz2
        
        conv_H2 = u * dH2_dx + v * dH2_dy + w * dH2_dz
        diff_species = (1.0 + nu_t / self.sc_t) / (self.re * self.sc / 1000.0)
        res_species = rho * conv_H2 - diff_species * lap_H2 + q_chem
        
        # 6. Post-Flame Radical Recombination Constraint: X_OH -> 0 for z > 40 mm
        mask_downstream = (x_pde[:, 2:3] > (0.040 / config.L_0))
        if mask_downstream.any():
            loss_recomb = torch.mean(outputs['X_OH'][mask_downstream] ** 2)
        else:
            loss_recomb = torch.tensor(0.0, device=x_pde.device)
        
        return {
            'loss_cont': torch.mean(res_cont ** 2),
            'loss_mom': torch.mean(res_mom_x ** 2) + torch.mean(res_mom_y ** 2) + torch.mean(res_mom_z ** 2),
            'loss_swirl': torch.mean(res_swirl ** 2),
            'loss_energy': torch.mean(res_energy ** 2),
            'loss_species': torch.mean(res_species ** 2),
            'loss_recomb': loss_recomb
        }

    def boundary_residuals(self, model, bc_dict):
        """
        Evaluates boundary conditions on boundary collocation points.
        """
        # 1. Swirler Inlet (z = 0, annular injector)
        in_pts = bc_dict['inlet']
        out_in = model(in_pts)
        phi_in = in_pts[:, 3:4]
        w_target = 0.5 + 0.6 * (phi_in - 0.55) / 0.45
        loss_in_vel = torch.mean((out_in['w'] - w_target)**2)
        loss_in_temp = torch.mean((out_in['T'] - 0.0)**2)
        
        # Swirl angle target: V_theta = w * tan(45 deg) = w
        x_in = in_pts[:, 0:1]
        y_in = in_pts[:, 1:2]
        r_in = torch.sqrt(x_in**2 + y_in**2 + 1e-4)
        v_th_in = (x_in * out_in['v'] - y_in * out_in['u']) / r_in
        loss_in_swirl = torch.mean((v_th_in - out_in['w'])**2)
        
        # 2. Dump Wall & Liner Walls (No-slip: u = v = w = 0)
        dump_pts = bc_dict['dump']
        out_dump = model(dump_pts)
        loss_dump_noslip = torch.mean(out_dump['u']**2 + out_dump['v']**2 + out_dump['w']**2)
        
        wall_pts = bc_dict['wall']
        out_wall = model(wall_pts)
        loss_wall_noslip = torch.mean(out_wall['u']**2 + out_wall['v']**2 + out_wall['w']**2)
        
        # 3. Combustor Outlet (p = 0)
        out_pts = bc_dict['outlet']
        out_outlet = model(out_pts)
        loss_out_p = torch.mean(out_outlet['p']**2)
        
        loss_bc = (loss_in_vel + loss_in_temp + loss_in_swirl + 
                   loss_dump_noslip + loss_wall_noslip + loss_out_p)
        return loss_bc

    def data_residual(self, model, X_data, Y_data):
        """
        Backward-compatible 2-variable data residual.
        """
        out = model(X_data)
        w_pred = out['w']
        T_pred = out['T']
        loss_w = torch.mean((w_pred - Y_data[:, 0:1])**2)
        loss_T = torch.mean((T_pred - Y_data[:, 1:2])**2)
        return loss_w + loss_T

    def data_residual_multiphysics(self, model, X_rake, Y_rake, X_wall, Y_wall, global_dict=None):
        """
        Supervised data anchor loss matching all multi-physics variables from ANSYS Fluent 2025 R2:
        - Axial velocity Wz
        - Tangential swirl velocity V_theta
        - Static pressure p
        - Static temperature T
        - OH radical mole fraction X_OH
        - Liner wall heat flux q_wall
        - Combustor aerodynamic pressure drop Delta_p across training throttles
        - Global outlet NO emissions
        """
        out_rake = model(X_rake)
        w_pred = out_rake['w']
        v_pred = out_rake['v']
        p_pred = out_rake['p']
        T_pred = out_rake['T']
        oh_pred = out_rake['X_OH']
        
        loss_w = torch.mean((w_pred - Y_rake[:, 0:1]) ** 2)
        loss_v = torch.mean((v_pred - Y_rake[:, 1:2]) ** 2)
        loss_p = torch.mean((p_pred - Y_rake[:, 2:3]) ** 2)
        loss_T = torch.mean((T_pred - Y_rake[:, 3:4]) ** 2)
        loss_oh = torch.mean((oh_pred - Y_rake[:, 4:5]) ** 2)
        
        # Wall heat flux loss (elevated priority to resolve boundary layer heat transfer)
        out_wall = model(X_wall)
        q_pred = out_wall['q_wall']
        loss_q = torch.mean((q_pred - Y_wall) ** 2)
        
        # Macro combustor pressure drop constraint across training throttles (Delta_p = p_in - p_out)
        loss_pdrop = torch.tensor(0.0, device=X_rake.device)
        loss_global = torch.tensor(0.0, device=X_rake.device)
        if global_dict is not None and len(global_dict) > 0:
            for phi_val, gvals in global_dict.items():
                pdrop_target = gvals['pdrop'] / config.P_REF
                
                # Sample annular inlet points (z = 0, r in [R_hub, R_tip])
                r_inlet = torch.linspace(config.R_HUB / config.L_0, config.R_TIP / config.L_0, 16, device=X_rake.device)
                x_in = torch.stack([r_inlet, torch.zeros_like(r_inlet), torch.zeros_like(r_inlet), torch.full_like(r_inlet, phi_val)], dim=-1)
                p_in_pred = model(x_in)['p']
                
                # Sample outlet points (z = L_chamber)
                r_out = torch.linspace(0.0, config.R_CHAMBER / config.L_0, 16, device=X_rake.device)
                x_out = torch.stack([r_out, torch.zeros_like(r_out), torch.full_like(r_out, config.L_CHAMBER / config.L_0), torch.full_like(r_out, phi_val)], dim=-1)
                p_out_pred = model(x_out)['p']
                
                pdrop_pred = torch.mean(p_in_pred) - torch.mean(p_out_pred)
                loss_pdrop = loss_pdrop + (pdrop_pred - pdrop_target) ** 2
                
                # Global NOx emissions target
                phi_t = torch.tensor([[0.0, 0.0, config.L_CHAMBER / config.L_0, phi_val]], device=X_rake.device, dtype=config.TORCH_DTYPE)
                out_outlet = model(phi_t)
                target_no = gvals['nox'] / config.NO_REF
                loss_global = loss_global + (out_outlet['X_NO'][0, 0] - target_no) ** 2
                
            loss_pdrop = loss_pdrop / len(global_dict)
            loss_global = loss_global / len(global_dict)
        
        total_data_loss = (loss_w + loss_v + 1.5 * loss_p + loss_T + 1.5 * loss_oh + 
                           2.0 * loss_q + 1.5 * loss_pdrop + 0.1 * loss_global)
        return {
            'loss_data_total': total_data_loss,
            'loss_w': loss_w,
            'loss_v': loss_v,
            'loss_p': loss_p,
            'loss_T': loss_T,
            'loss_oh': loss_oh,
            'loss_q': loss_q,
            'loss_pdrop': loss_pdrop,
            'loss_global': loss_global
        }

if __name__ == "__main__":
    from network import FourierResPINN
    from collocation_sampler import CombustorCollocationSampler
    
    net = FourierResPINN()
    loss_engine = CombustorPhysicsLoss()
    sampler = CombustorCollocationSampler()
    
    x_test = sampler.sample_interior_points(256)
    bc_test = sampler.sample_boundary_points(128)
    x_rake = torch.randn(64, 4)
    y_rake = torch.randn(64, 5)
    x_wall = torch.randn(32, 4)
    y_wall = torch.randn(32, 1)
    
    print("--- Multi-Physics Loss Engine Verification ---")
    pde_losses = loss_engine.pde_residuals(net, x_test)
    bc_loss = loss_engine.boundary_residuals(net, bc_test)
    data_dict = loss_engine.data_residual_multiphysics(net, x_rake, y_rake, x_wall, y_wall)
    
    for k, v in pde_losses.items():
        print(f"  {k:<16}: {v.item():.6f}")
    print(f"  loss_bc         : {bc_loss.item():.6f}")
    for k, v in data_dict.items():
        print(f"  {k:<16}: {v.item():.6f}")
    print("All multi-physics autograd and data residuals evaluated successfully!")

