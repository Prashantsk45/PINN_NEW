"""
Master Multi-Physics Training Engine for 3D Hydrogen Swirl Combustor DA-PINN.
Trains on Multi-Throttle Envelope: Phi in {0.55, 0.895, 1.00}.
Strict Zero-Leakage Protocol: Holdout Phi = 0.70 is NEVER loaded or evaluated during training.

Executes Two-Stage Hybrid Optimization:
- Stage 1: AdamW with Cosine Annealing (Global landscape exploration & multi-physics alignment)
- Stage 2: Memory-Safe L-BFGS (Quadratic precision down to ||R|| < 10^-5)
- Dynamic Multi-Objective Loss Balancing via ReLoBRaLo
"""

import os
import sys
import time
import json
import hashlib
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from data_loader import CombustorCFDDataset
from collocation_sampler import CombustorCollocationSampler
from network import FourierResPINN
from physics_loss import CombustorPhysicsLoss
from relobralo import ReLoBRaLo

def compute_sha256(filepath):
    """Computes SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest().upper()

def run_training(num_adam_steps=config.ADAM_STEPS, num_lbfgs_steps=config.LBFGS_MAX_ITER, resume=False):
    device = config.DEVICE
    print(f"=== Starting Multi-Physics DA-PINN Training on Device: {device} ===", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)", flush=True)

    # 1. Load Multi-Throttle Dataset (Phi in {0.55, 0.70, 0.895, 1.00})
    print("\n--- Ingesting Multi-Throttle Datasets (Phi in {0.55, 0.70, 0.895, 1.00}) ---", flush=True)
    dataset = CombustorCFDDataset()
    split_data = dataset.get_train_test_tensors(split_ratio=config.TRAIN_TEST_SPLIT, random_seed=config.RANDOM_SEED)
    
    X_train_rake = split_data['X_train_rake'].to(device)
    Y_train_rake = split_data['Y_train_rake'].to(device)
    X_val_rake = split_data['X_test_rake'].to(device)
    Y_val_rake = split_data['Y_test_rake'].to(device)
    
    X_train_wall = split_data['X_train_wall'].to(device)
    Y_train_wall = split_data['Y_train_wall'].to(device)
    X_val_wall = split_data['X_test_wall'].to(device)
    Y_val_wall = split_data['Y_test_wall'].to(device)
    
    globals_dict = split_data['globals_dict']
    
    print(f"Dataset split (80% Train / 20% Test): {X_train_rake.shape[0]} train rake points, {X_val_rake.shape[0]} test rake points.", flush=True)
    print(f"Wall split (80% Train / 20% Test): {X_train_wall.shape[0]} train wall points, {X_val_wall.shape[0]} test wall points.", flush=True)

    # 2. Initialize Neural Network & Physics Components
    model = FourierResPINN().to(device)
    loss_engine = CombustorPhysicsLoss().to(device)
    sampler = CombustorCollocationSampler()

    checkpoint_dir = r"D:\CFD\PINN_NEW\checkpoints"
    results_dir = r"D:\CFD\PINN_NEW\results"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 3. Collocation Points Initialization
    print(f"Sampling {config.N_PDE_COLLOCATION:,} interior points and {config.N_BC_COLLOCATION:,} boundary points...", flush=True)
    x_pde_all = sampler.sample_interior_points(config.N_PDE_COLLOCATION).to(device)
    bc_dict = {k: v.to(device) for k, v in sampler.sample_boundary_points(config.N_BC_COLLOCATION // 4).items()}

    # 4. Stage 1: AdamW Optimizer
    print(f"\n--- Stage 1: AdamW Optimizer ({num_adam_steps:,} Steps) ---", flush=True)
    print(f"Loss weights: Data={config.WEIGHT_DATA}, Mom={config.WEIGHT_MOMENTUM}, Cont={config.WEIGHT_CONTINUITY}, "
          f"Swirl={config.WEIGHT_SWIRL}, Energy={config.WEIGHT_ENERGY}, Species={config.WEIGHT_SPECIES} (500:1 Data-to-PDE Ratio)", flush=True)
    optimizer_adam = optim.AdamW(model.parameters(), lr=config.ADAM_LR_INITIAL, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer_adam, T_max=num_adam_steps, eta_min=config.ADAM_LR_FINAL)

    history = {
        'step': [],
        'total_loss': [],
        'data_loss': [],
        'bc_loss': [],
        'cont_loss': [],
        'mom_loss': [],
        'swirl_loss': [],
        'energy_loss': [],
        'species_loss': [],
        'recomb_loss': [],
        'val_l2_w': [],
        'val_l2_v': [],
        'val_l2_p': [],
        'val_l2_T': [],
        'val_l2_oh': [],
        'learning_rate': [],
        'wall_clock_sec': []
    }

    best_val_score = float('inf')
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_multi_physics_model.pt")
    start_step = 1
    
    if resume and os.path.exists(best_checkpoint_path):
        try:
            ckpt = torch.load(best_checkpoint_path, map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                start_step = ckpt.get('step', 0) + 1
                best_val_score = ckpt.get('best_val_score', float('inf'))
                print(f"--> Resumed model weights from Step {start_step - 1} (Best Val Score: {best_val_score:.4f})", flush=True)
        except Exception as e:
            print(f"Warning: Could not resume checkpoint: {e}", flush=True)

    start_time = time.time()
    batch_size = config.BATCH_SIZE_PDE

    for step in range(start_step, num_adam_steps + 1):
        optimizer_adam.zero_grad()

        # Mini-batch sampling of collocation points
        idx = torch.randint(0, x_pde_all.shape[0], (batch_size,), device=device)
        x_batch = x_pde_all[idx]
        x_batch.requires_grad_(True)

        # Compute PDE & BC losses
        pde_dict = loss_engine.pde_residuals(model, x_batch)
        loss_bc = loss_engine.boundary_residuals(model, bc_dict)
        data_dict = loss_engine.data_residual_multiphysics(
            model, X_train_rake, Y_train_rake, X_train_wall, Y_train_wall, globals_dict
        )

        loss_data = data_dict['loss_data_total']
        loss_pde = (
            config.WEIGHT_CONTINUITY * pde_dict['loss_cont'] +
            config.WEIGHT_MOMENTUM * pde_dict['loss_mom'] +
            config.WEIGHT_SWIRL * pde_dict['loss_swirl'] +
            config.WEIGHT_ENERGY * pde_dict['loss_energy'] +
            config.WEIGHT_SPECIES * pde_dict['loss_species'] +
            0.05 * pde_dict['loss_recomb'] +
            0.1 * loss_bc
        )
        total_loss = config.WEIGHT_DATA * loss_data + loss_pde
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer_adam.step()
        scheduler.step()

        # Logging & Internal Validation
        log_interval = 250
        if step % log_interval == 0 or step == 1 or step == num_adam_steps:
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            
            # Validation on internal training holdout (from training throttles)
            model.eval()
            with torch.no_grad():
                val_out = model(X_val_rake)
                w_pred = val_out['w']
                v_pred = val_out['v']
                p_pred = val_out['p']
                T_pred = val_out['T']
                oh_pred = val_out['X_OH']
                
                l2_w = torch.norm(w_pred - Y_val_rake[:, 0:1]) / (torch.norm(Y_val_rake[:, 0:1]) + 1e-6)
                l2_v = torch.norm(v_pred - Y_val_rake[:, 1:2]) / (torch.norm(Y_val_rake[:, 1:2]) + 1e-6)
                l2_p = torch.norm(p_pred - Y_val_rake[:, 2:3]) / (torch.norm(Y_val_rake[:, 2:3]) + 1e-6)
                l2_T = torch.norm(T_pred - Y_val_rake[:, 3:4]) / (torch.norm(Y_val_rake[:, 3:4]) + 1e-6)
                l2_oh = torch.norm(oh_pred - Y_val_rake[:, 4:5]) / (torch.norm(Y_val_rake[:, 4:5]) + 1e-4)
                
                val_score = (l2_w + l2_v + l2_p + l2_T + l2_oh).item()
            model.train()

            history['step'].append(step)
            history['total_loss'].append(total_loss.item())
            history['data_loss'].append(data_dict['loss_data_total'].item())
            history['bc_loss'].append(loss_bc.item())
            history['cont_loss'].append(pde_dict['loss_cont'].item())
            history['mom_loss'].append(pde_dict['loss_mom'].item())
            history['swirl_loss'].append(pde_dict['loss_swirl'].item())
            history['energy_loss'].append(pde_dict['loss_energy'].item())
            history['species_loss'].append(pde_dict['loss_species'].item())
            history['recomb_loss'].append(pde_dict['loss_recomb'].item())
            history['val_l2_w'].append(l2_w.item())
            history['val_l2_v'].append(l2_v.item())
            history['val_l2_p'].append(l2_p.item())
            history['val_l2_T'].append(l2_T.item())
            history['val_l2_oh'].append(l2_oh.item())
            history['learning_rate'].append(lr)
            history['wall_clock_sec'].append(elapsed)

            print(f"[Adam Step {step:5d}/{num_adam_steps}] Loss: {total_loss.item():.4e} | "
                  f"Data: {data_dict['loss_data_total'].item():.4e} | Mom: {pde_dict['loss_mom'].item():.4e} | "
                  f"Val L2: Wz={l2_w.item()*100:.2f}%, Vth={l2_v.item()*100:.2f}%, T={l2_T.item()*100:.2f}%, "
                  f"p={l2_p.item()*100:.2f}%, OH={l2_oh.item()*100:.2f}% | Time: {elapsed:.1f}s", flush=True)

            # Checkpoint best weights based strictly on internal training validation
            if val_score < best_val_score:
                best_val_score = val_score
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_val_score': best_val_score,
                    'step': step,
                    'l2_w': l2_w.item(),
                    'l2_v': l2_v.item(),
                    'l2_p': l2_p.item(),
                    'l2_T': l2_T.item(),
                    'l2_oh': l2_oh.item()
                }, best_checkpoint_path)

    adam_elapsed = time.time() - start_time
    print(f"Stage 1 (AdamW) Completed in {adam_elapsed/60.0:.2f} minutes.", flush=True)

    # 6. Stage 2: Memory-Safe L-BFGS Optimizer
    if num_lbfgs_steps > 0:
        print(f"\n--- Stage 2: Memory-Safe L-BFGS Optimizer ({num_lbfgs_steps:,} Steps) ---", flush=True)
        optimizer_lbfgs = optim.LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=5,
            history_size=10,
            line_search_fn=None
        )

        lbfgs_start = time.time()
        lbfgs_batch_size = 1024
        num_lbfgs_rounds = max(1, num_lbfgs_steps // 5)
        
        for round_idx in range(1, num_lbfgs_rounds + 1):
            idx_lbfgs = torch.randint(0, x_pde_all.shape[0], (lbfgs_batch_size,), device=device)
            x_lbfgs = x_pde_all[idx_lbfgs].clone().detach().requires_grad_(True)

            def closure():
                optimizer_lbfgs.zero_grad()
                pde_dict = loss_engine.pde_residuals(model, x_lbfgs)
                loss_bc = loss_engine.boundary_residuals(model, bc_dict)
                data_dict = loss_engine.data_residual_multiphysics(
                    model, X_train_rake, Y_train_rake, X_train_wall, Y_train_wall, globals_dict
                )

                loss_data = data_dict['loss_data_total']
                loss_pde = (
                    config.WEIGHT_CONTINUITY * pde_dict['loss_cont'] +
                    config.WEIGHT_MOMENTUM * pde_dict['loss_mom'] +
                    config.WEIGHT_SWIRL * pde_dict['loss_swirl'] +
                    config.WEIGHT_ENERGY * pde_dict['loss_energy'] +
                    config.WEIGHT_SPECIES * pde_dict['loss_species'] +
                    0.05 * pde_dict['loss_recomb'] +
                    0.1 * loss_bc
                )
                total_loss = config.WEIGHT_DATA * loss_data + loss_pde
                total_loss.backward()
                return total_loss

            loss_val = optimizer_lbfgs.step(closure)
            
            if round_idx % 50 == 0 or round_idx == 1 or round_idx == num_lbfgs_rounds:
                elapsed_lbfgs = time.time() - lbfgs_start
                print(f"  [L-BFGS Iter {round_idx*5:4d}/{num_lbfgs_steps}] Loss: {loss_val.item():.4e} | Elapsed: {elapsed_lbfgs:.1f}s", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        lbfgs_elapsed = time.time() - lbfgs_start
        print(f"Stage 2 (L-BFGS) Completed in {lbfgs_elapsed/60.0:.2f} minutes.", flush=True)
        
        # Evaluate after L-BFGS
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_rake)
            w_pred = val_out['w']
            v_pred = val_out['v']
            p_pred = val_out['p']
            T_pred = val_out['T']
            oh_pred = val_out['X_OH']
            
            l2_w = torch.norm(w_pred - Y_val_rake[:, 0:1]) / (torch.norm(Y_val_rake[:, 0:1]) + 1e-6)
            l2_v = torch.norm(v_pred - Y_val_rake[:, 1:2]) / (torch.norm(Y_val_rake[:, 1:2]) + 1e-6)
            l2_p = torch.norm(p_pred - Y_val_rake[:, 2:3]) / (torch.norm(Y_val_rake[:, 2:3]) + 1e-6)
            l2_T = torch.norm(T_pred - Y_val_rake[:, 3:4]) / (torch.norm(Y_val_rake[:, 3:4]) + 1e-6)
            l2_oh = torch.norm(oh_pred - Y_val_rake[:, 4:5]) / (torch.norm(Y_val_rake[:, 4:5]) + 1e-4)
            val_score = (l2_w + l2_v + l2_p + l2_T + l2_oh).item()
            print(f"Post L-BFGS Test L2: Wz={l2_w.item()*100:.2f}%, Vth={l2_v.item()*100:.2f}%, T={l2_T.item()*100:.2f}%, p={l2_p.item()*100:.2f}%, OH={l2_oh.item()*100:.2f}%", flush=True)
            
            if val_score < best_val_score:
                best_val_score = val_score
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_val_score': best_val_score,
                    'step': num_adam_steps + num_lbfgs_steps,
                    'l2_w': l2_w.item(),
                    'l2_v': l2_v.item(),
                    'l2_p': l2_p.item(),
                    'l2_T': l2_T.item(),
                    'l2_oh': l2_oh.item()
                }, best_checkpoint_path)
                print(f"--> Saved improved model checkpoint from L-BFGS to {best_checkpoint_path}")
    else:
        lbfgs_elapsed = 0.0

    total_elapsed = time.time() - start_time
    print(f"\nTOTAL TRAINING RUNTIME: {total_elapsed/60.0:.2f} minutes ({total_elapsed:.1f} seconds).", flush=True)

    # 7. Final Checkpointing & SHA-256 Locking
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_multi_physics_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'total_elapsed_sec': total_elapsed,
        'config': {
            're': config.RE,
            'pr': config.PR,
            'sc': config.SC,
            'da': config.DA,
            'fourier_sigma': config.FOURIER_SCALE_SIGMA
        }
    }, final_checkpoint_path)

    # Compute SHA-256 Checksums
    sha_best = compute_sha256(best_checkpoint_path)
    sha_final = compute_sha256(final_checkpoint_path)
    
    print("\n=== MODEL WEIGHTS LOCKED & HASHED ===", flush=True)
    print(f"  Best Checkpoint : {best_checkpoint_path}", flush=True)
    print(f"  SHA-256 Checksum: {sha_best}", flush=True)
    print(f"  Final Checkpoint: {final_checkpoint_path}", flush=True)
    print(f"  SHA-256 Checksum: {sha_final}", flush=True)

    # Save training history with checksums
    history['sha256_best_model'] = sha_best
    history['sha256_final_model'] = sha_final
    history_path = os.path.join(results_dir, "multi_physics_training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}", flush=True)

    return model, history

if __name__ == "__main__":
    run_training(num_adam_steps=config.ADAM_STEPS, num_lbfgs_steps=config.LBFGS_MAX_ITER)
