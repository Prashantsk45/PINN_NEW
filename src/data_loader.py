"""
Data loader and preprocessor for Physics-Informed Neural Network (DA-PINN).
Parses ANSYS Fluent 2025 R2 CFD results (.xy rakes) and experimental benchmark data.
Splits into Training Anchors (Idle, Cruise, Takeoff) and Blind Holdout Test (Approach).
"""

import os
import re
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import config

def parse_xy_file(filepath):
    """
    Parses an ANSYS Fluent .xy curve export file.
    Returns a dictionary mapping station label -> list of (radial_pos_m, value).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CFD file not found: {filepath}")
        
    profiles = {}
    current_label = None
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "xy/key/label" in line:
                # Extract label, e.g. "line-z-10"
                match = re.search(r'"([^"]+)"', line)
                if match:
                    current_label = match.group(1)
                    profiles[current_label] = []
            elif current_label and line and not line.startswith("("):
                parts = line.split()
                if len(parts) == 2:
                    try:
                        pos = float(parts[0])   # Radial position [m]
                        val = float(parts[1])   # Physical value (m/s or K)
                        profiles[current_label].append((pos, val))
                    except ValueError:
                        pass
                        
    return profiles

def parse_scalar_report(filepath):
    """
    Parses an ANSYS Fluent Surface/Volume Integral Report.
    Extracts the net/outlet scalar quantity.
    """
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    if nums:
        return float(nums[-1])
    return None

def label_to_z_coord(label):
    """
    Converts line label (e.g. 'line-z-25') to axial z-coordinate in meters (0.025 m).
    """
    match = re.search(r'line-z-(\d+)', label)
    if match:
        z_mm = float(match.group(1))
        return z_mm * 1e-3
    return None

class CombustorCFDDataset:
    """
    Multi-Physics CFD Dataset Loader for 3D Swirl Hydrogen Combustor.
    Loads 5 rake variables (Wz, V_theta, p, T, X_OH), liner wall heat flux (q_wall),
    and global aerothermodynamic scalars (Tmax, Delta_p, NOx, Tout)
    across the training envelope: Phi in {0.55, 0.895, 1.00}.
    """
    def __init__(self, train_dirs=config.TRAIN_DATA_DIRS, exp_dir=config.EXP_DIR):
        self.train_dirs = train_dirs
        self.exp_dir = exp_dir
        self.training_records = []
        self.wall_records = []
        self.global_metrics = {}
        self.load_training_data()

    def load_training_data(self):
        """
        Loads training datasets strictly from Phi in {0.55, 0.895, 1.00}.
        Zero leakage: Holdout Phi = 0.70 is strictly NOT loaded here.
        """
        for phi, folder in self.train_dirs.items():
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Training directory not found: {folder}")
            
            # 1. Identify Rake Files
            vel_f = [f for f in os.listdir(folder) if "axial_velocity" in f][0]
            temp_f = [f for f in os.listdir(folder) if "temperature" in f][0]
            
            p_w = parse_xy_file(os.path.join(folder, vel_f))
            p_v = parse_xy_file(os.path.join(folder, "TANGENTIAL VELOCITY"))
            p_p = parse_xy_file(os.path.join(folder, "STATIC PRESSURE"))
            p_t = parse_xy_file(os.path.join(folder, temp_f))
            p_oh = parse_xy_file(os.path.join(folder, "OH CONCENTRATION"))
            
            # Combine 5 variables across 7 rakes
            for label, w_pts in p_w.items():
                z_m = label_to_z_coord(label)
                if z_m is None:
                    continue
                v_dict = {round(pt[0], 6): pt[1] for pt in p_v.get(label, [])}
                p_dict = {round(pt[0], 6): pt[1] for pt in p_p.get(label, [])}
                t_dict = {round(pt[0], 6): pt[1] for pt in p_t.get(label, [])}
                oh_dict = {round(pt[0], 6): pt[1] for pt in p_oh.get(label, [])}
                
                for r_m, w_z in w_pts:
                    rk = round(r_m, 6)
                    v_th = v_dict.get(rk, 0.0)
                    p_stat = p_dict.get(rk, 0.0)
                    t_k = t_dict.get(rk, config.T_0)
                    x_oh = oh_dict.get(rk, 0.0)
                    
                    self.training_records.append({
                        'x_m': r_m,
                        'y_m': 0.0,
                        'z_m': z_m,
                        'phi': phi,
                        'w_z': w_z,
                        'v_th': v_th,
                        'p': p_stat,
                        'T': t_k,
                        'X_OH': x_oh
                    })
            
            # 2. Wall Heat Flux
            whf_path = os.path.join(folder, "WALL HEAT FLUX")
            if os.path.exists(whf_path):
                p_whf = parse_xy_file(whf_path)
                wall_pts = p_whf.get('combustor-walls', [])
                # Downsample to ~500 evenly spaced axial positions along wall
                if len(wall_pts) > 500:
                    step = len(wall_pts) // 500
                    wall_pts = wall_pts[::step]
                for z_w, q_w in wall_pts:
                    self.wall_records.append({
                        'x_m': config.R_CHAMBER,
                        'y_m': 0.0,
                        'z_m': z_w,
                        'phi': phi,
                        'q_wall': q_w
                    })
            
            # 3. Global Scalar Reports
            nox = parse_scalar_report(os.path.join(folder, "Exit NOx"))
            tmax = parse_scalar_report(os.path.join(folder, "Peak Flame Temperature"))
            pdrop = parse_scalar_report(os.path.join(folder, "Pressure Drop"))
            tout = parse_scalar_report(os.path.join(folder, "Outlet Temperature"))
            self.global_metrics[phi] = {
                'nox': nox if nox is not None else 1e-4,
                'tmax': tmax if tmax is not None else 2400.0,
                'pdrop': pdrop if pdrop is not None else 1300.0,
                'tout': tout if tout is not None else 2000.0
            }
            
            print(f"Loaded Training Throttle Phi = {phi:.3f}: {len(p_w)} rakes, "
                  f"Exit NOx={self.global_metrics[phi]['nox']:.2e}, "
                  f"Tmax={self.global_metrics[phi]['tmax']:.1f} K, "
                  f"Pdrop={self.global_metrics[phi]['pdrop']:.1f} Pa.")

    def get_training_tensors(self):
        """
        Returns normalized training tensors.
        Returns:
            X_rake: (N, 4) tensor [x_tilde, y_tilde, z_tilde, phi]
            Y_rake: (N, 5) tensor [w_tilde, v_tilde, p_tilde, T_tilde, X_OH_tilde]
            X_wall: (M, 4) tensor [x_tilde, y_tilde, z_tilde, phi]
            Y_wall: (M, 1) tensor [q_wall_tilde]
            global_metrics: dict mapping phi -> scalars
        """
        df_rake = pd.DataFrame(self.training_records)
        df_wall = pd.DataFrame(self.wall_records)
        
        # 1. Normalize Rake Coordinates & Targets
        xr = df_rake['x_m'].values / config.L_0
        yr = df_rake['y_m'].values / config.L_0
        zr = df_rake['z_m'].values / config.L_0
        phir = df_rake['phi'].values
        X_rake = np.stack([xr, yr, zr, phir], axis=1)
        
        w_t = df_rake['w_z'].values / config.U_0
        v_t = df_rake['v_th'].values / config.U_0
        p_t = df_rake['p'].values / config.P_REF
        T_t = (df_rake['T'].values - config.T_0) / (config.T_AD - config.T_0)
        oh_t = df_rake['X_OH'].values / config.OH_REF
        Y_rake = np.stack([w_t, v_t, p_t, T_t, oh_t], axis=1)
        
        # 2. Normalize Wall Coordinates & Targets
        xw = df_wall['x_m'].values / config.L_0
        yw = df_wall['y_m'].values / config.L_0
        zw = df_wall['z_m'].values / config.L_0
        phiw = df_wall['phi'].values
        X_wall = np.stack([xw, yw, zw, phiw], axis=1)
        
        qw_t = df_wall['q_wall'].values / config.Q_WALL_REF
        Y_wall = qw_t[:, None]
        
        return (torch.tensor(X_rake, dtype=config.TORCH_DTYPE),
                torch.tensor(Y_rake, dtype=config.TORCH_DTYPE),
                torch.tensor(X_wall, dtype=config.TORCH_DTYPE),
                torch.tensor(Y_wall, dtype=config.TORCH_DTYPE),
                self.global_metrics)

    def get_train_test_tensors(self, split_ratio=config.TRAIN_TEST_SPLIT, random_seed=config.RANDOM_SEED):
        """
        Splits multi-throttle rake and wall data into Train (80%) and Test (20%) sets.
        Matches exact methodology of D:\CFD\PINN.
        """
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        X_rake, Y_rake, X_wall, Y_wall, globals_dict = self.get_training_tensors()
        
        # Rake split
        n_rake = X_rake.shape[0]
        perm_rake = torch.randperm(n_rake)
        n_train_rake = int(split_ratio * n_rake)
        
        train_rake_idx = perm_rake[:n_train_rake]
        test_rake_idx = perm_rake[n_train_rake:]
        
        X_train_rake = X_rake[train_rake_idx]
        Y_train_rake = Y_rake[train_rake_idx]
        X_test_rake = X_rake[test_rake_idx]
        Y_test_rake = Y_rake[test_rake_idx]
        
        # Wall split
        n_wall = X_wall.shape[0]
        perm_wall = torch.randperm(n_wall)
        n_train_wall = int(split_ratio * n_wall)
        
        train_wall_idx = perm_wall[:n_train_wall]
        test_wall_idx = perm_wall[n_train_wall:]
        
        X_train_wall = X_wall[train_wall_idx]
        Y_train_wall = Y_wall[train_wall_idx]
        X_test_wall = X_wall[test_wall_idx]
        Y_test_wall = Y_wall[test_wall_idx]
        
        return {
            'X_train_rake': X_train_rake,
            'Y_train_rake': Y_train_rake,
            'X_test_rake': X_test_rake,
            'Y_test_rake': Y_test_rake,
            'X_train_wall': X_train_wall,
            'Y_train_wall': Y_train_wall,
            'X_test_wall': X_test_wall,
            'Y_test_wall': Y_test_wall,
            'train_rake_idx': train_rake_idx,
            'test_rake_idx': test_rake_idx,
            'globals_dict': globals_dict
        }

    def load_holdout_benchmark_data(self, holdout_dir=config.HOLDOUT_DATA_DIR):
        """
        Post-training evaluation ONLY. Loads Phi = 0.70 ground truth from CFD.
        """
        if not os.path.exists(holdout_dir):
            raise FileNotFoundError(f"Holdout directory not found: {holdout_dir}")
            
        vel_f = [f for f in os.listdir(holdout_dir) if "axial_velocity" in f][0]
        temp_f = [f for f in os.listdir(holdout_dir) if "temperature" in f][0]
        
        p_w = parse_xy_file(os.path.join(holdout_dir, vel_f))
        p_v = parse_xy_file(os.path.join(holdout_dir, "TANGENTIAL VELOCITY"))
        p_p = parse_xy_file(os.path.join(holdout_dir, "STATIC PRESSURE"))
        p_t = parse_xy_file(os.path.join(holdout_dir, temp_f))
        p_oh = parse_xy_file(os.path.join(holdout_dir, "OH CONCENTRATION"))
        p_whf = parse_xy_file(os.path.join(holdout_dir, "WALL HEAT FLUX"))
        
        records = []
        for label, w_pts in p_w.items():
            z_m = label_to_z_coord(label)
            if z_m is None:
                continue
            v_dict = {round(pt[0], 6): pt[1] for pt in p_v.get(label, [])}
            p_dict = {round(pt[0], 6): pt[1] for pt in p_p.get(label, [])}
            t_dict = {round(pt[0], 6): pt[1] for pt in p_t.get(label, [])}
            oh_dict = {round(pt[0], 6): pt[1] for pt in p_oh.get(label, [])}
            
            for r_m, w_z in w_pts:
                rk = round(r_m, 6)
                records.append({
                    'station': label,
                    'x_m': r_m,
                    'y_m': 0.0,
                    'z_m': z_m,
                    'r_m': abs(r_m),
                    'phi': 0.70,
                    'w_z': w_z,
                    'v_th': v_dict.get(rk, 0.0),
                    'p': p_dict.get(rk, 0.0),
                    'T': t_dict.get(rk, config.T_0),
                    'X_OH': oh_dict.get(rk, 0.0)
                })
                
        df_holdout = pd.DataFrame(records)
        
        whf_pts = p_whf.get('combustor-walls', [])
        df_whf = pd.DataFrame([{'z_m': p[0], 'q_wall': p[1], 'phi': 0.70} for p in whf_pts])
        
        globals_070 = {
            'nox': parse_scalar_report(os.path.join(holdout_dir, "Exit NOx")),
            'tmax': parse_scalar_report(os.path.join(holdout_dir, "Peak Flame Temperature")),
            'pdrop': parse_scalar_report(os.path.join(holdout_dir, "Pressure Drop")),
            'tout': parse_scalar_report(os.path.join(holdout_dir, "Outlet Temperature"))
        }
        return df_holdout, df_whf, globals_070


    def get_experimental_data(self):
        """
        Loads DLR CARS temperature and LDV velocity benchmarks.
        """
        cars_path = os.path.join(self.exp_dir, "DLR_H2_radial_temperature_z25_CARS.csv")
        ldv_path = os.path.join(self.exp_dir, "DLR_H2_radial_velocity_z5_LDV.csv")
        
        cars_df = pd.read_csv(cars_path) if os.path.exists(cars_path) else None
        ldv_df = pd.read_csv(ldv_path) if os.path.exists(ldv_path) else None
        
        return cars_df, ldv_df

if __name__ == "__main__":
    dataset = CombustorCFDDataset()
    X_rake, Y_rake, X_wall, Y_wall, globals_dict = dataset.get_training_tensors()
    
    print(f"\n--- Verification Summary ---")
    print(f"Training Rake Points:     {X_rake.shape[0]} samples (Shape: {X_rake.shape})")
    print(f"Training Rake Targets:    {Y_rake.shape[0]} samples (Shape: {Y_rake.shape})")
    print(f"Training Wall Points:     {X_wall.shape[0]} samples (Shape: {X_wall.shape})")
    print(f"Training Wall Targets:    {Y_wall.shape[0]} samples (Shape: {Y_wall.shape})")
    print(f"Global Metrics:           {globals_dict}")
    print(f"Non-dimensional X ranges: min={X_rake.min(0).values.numpy()}, max={X_rake.max(0).values.numpy()}")
    print(f"Non-dimensional Y ranges: min={Y_rake.min(0).values.numpy()}, max={Y_rake.max(0).values.numpy()}")

