"""
ReLoBRaLo: Relative Loss Balanced Residuals for Multi-Objective Physics-Informed Learning.
Paper: Bischof & Kraus (2022), Computer Methods in Applied Mechanics and Engineering.
Dynamically balances PDE, BC, and Data losses to prevent gradient starvation.
"""

import math
import torch
import torch.nn as nn

import config

class ReLoBRaLo:
    """
    Tracks loss components across iterations and dynamically balances weights
    based on relative rate of loss decrease with exponential moving average memory.
    """
    def __init__(self, 
                 loss_names=['data', 'bc', 'cont', 'mom', 'swirl', 'energy', 'species'],
                 temperature=config.RELOBRALO_TEMPERATURE,
                 alpha=config.RELOBRALO_ALPHA,
                 tau=config.RELOBRALO_TAU,
                 device=config.DEVICE):
        self.loss_names = loss_names
        self.num_losses = len(loss_names)
        self.temperature = temperature
        self.alpha = alpha
        self.tau = tau
        self.device = device
        
        # Initial uniform weights summing to num_losses
        self.lambdas = torch.ones(self.num_losses, device=self.device, dtype=torch.float32)
        
        # History of previous step losses and initial step losses
        self.prev_losses = None
        self.init_losses = None
        self.step_count = 0
        self.weight_history = {name: [] for name in loss_names}

    def update(self, current_loss_dict):
        """
        Updates dynamic weights based on current loss dictionary.
        Args:
            current_loss_dict: dict mapping loss_name -> scalar tensor
        Returns:
            dict mapping loss_name -> scalar weight float
            total_loss: scalar tensor of weighted sum
        """
        losses = torch.stack([current_loss_dict[name].detach() for name in self.loss_names])
        
        if self.step_count == 0:
            self.init_losses = losses.clone()
            self.prev_losses = losses.clone()
            self.step_count += 1
            for i, name in enumerate(self.loss_names):
                self.weight_history[name].append(self.lambdas[i].item())
            
            # Compute total weighted loss
            total_loss = sum(self.lambdas[i] * current_loss_dict[name] 
                             for i, name in enumerate(self.loss_names))
            return {name: self.lambdas[i].item() for i, name in enumerate(self.loss_names)}, total_loss

        # 1. Compute relative loss improvement ratio
        # rho_i = L_i(t) / (tau * L_i(t-1))
        eps = 1e-8
        rho = losses / (self.tau * self.prev_losses + eps)
        
        # 2. Softmax normalization over loss terms
        rho_scaled = rho / self.temperature
        # Subtract max for numerical stability in softmax
        rho_exp = torch.exp(rho_scaled - torch.max(rho_scaled))
        lambda_hat = self.num_losses * (rho_exp / (torch.sum(rho_exp) + eps))
        
        # 3. Exponential moving average update
        self.lambdas = self.alpha * self.lambdas + (1.0 - self.alpha) * lambda_hat
        
        # Clamp weights to prevent extreme values [0.05, 50.0]
        self.lambdas = torch.clamp(self.lambdas, min=0.05, max=50.0)
        
        # Update memory
        self.prev_losses = losses.clone()
        self.step_count += 1
        
        for i, name in enumerate(self.loss_names):
            self.weight_history[name].append(self.lambdas[i].item())
            
        # Compute total weighted loss
        total_loss = sum(self.lambdas[i] * current_loss_dict[name] 
                         for i, name in enumerate(self.loss_names))
        
        weight_dict = {name: self.lambdas[i].item() for i, name in enumerate(self.loss_names)}
        return weight_dict, total_loss

if __name__ == "__main__":
    balancer = ReLoBRaLo(device=torch.device("cpu"))
    print("--- ReLoBRaLo Balancer Verification ---")
    dummy_losses = {
        'data': torch.tensor(2.0),
        'bc': torch.tensor(1.0),
        'cont': torch.tensor(3.5),
        'mom': torch.tensor(4.0),
        'swirl': torch.tensor(1.2),
        'energy': torch.tensor(0.08),
        'species': torch.tensor(0.05)
    }
    
    weights, total = balancer.update(dummy_losses)
    print("Step 0 (Initial Weights):", weights)
    print(f"Total Weighted Loss: {total.item():.4f}")
    
    # Simulate step 1 with species decreasing slower than boundary
    dummy_losses['bc'] *= 0.5      # BC dropped fast
    dummy_losses['species'] *= 0.98 # Species dropped slow
    weights, total = balancer.update(dummy_losses)
    print("Step 1 (Updated Weights):", weights)
    print(f"Total Weighted Loss: {total.item():.4f}")
    print("ReLoBRaLo dynamically adjusts weights to prevent slower-converging physics from starving!")
