"""
Configuration file for Physics-Informed Neural Network (DA-PINN) Surrogate
for 3D Hydrogen Swirl Aero-Engine Combustor Aerothermodynamics.

Matches exact ANSYS Fluent 2025 R2 CFD domain:
Combustor Diameter D = 85 mm (Radius R = 42.5 mm), Length L = 110 mm.
"""

import math
import torch

# ==============================================================================
# 1. HARDWARE & DEVICE
# ==============================================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float32

# ==============================================================================
# 2. COMBUSTOR GEOMETRY (METRIC UNITS [m])
# ==============================================================================
R_CHAMBER = 0.0425          # Chamber outer radius: 42.5 mm (Diameter 85 mm)
D_CHAMBER = 2.0 * R_CHAMBER # Chamber diameter: 85 mm
L_CHAMBER = 0.110           # Chamber length: 110 mm

R_TIP = 0.0150              # Swirler outer tip radius: 15 mm (Diameter 30 mm)
R_HUB = 0.0060              # Swirler center hub radius: 6 mm (Diameter 12 mm)
SWIRL_ANGLE_DEG = 45.0      # Vane angle: 45 degrees
SWIRL_NUMBER_SG = 0.85      # Geometric swirl number: S_g = 0.85

# 7 Standard Axial Measurement Stations [m]
AXIAL_STATIONS = [0.005, 0.010, 0.015, 0.025, 0.040, 0.060, 0.080]

# ==============================================================================
# 3. CHARACTERISTIC SCALES & NON-DIMENSIONALIZATION (Mao & Karniadakis 2023)
# ==============================================================================
L_0 = 0.085                 # Length scale [m]
U_0 = 151.54                # Peak velocity scale at Cruise [m/s]
T_0 = 300.0                 # Ambient inlet temperature [K]
T_AD = 2380.0               # Adiabatic flame temperature at Cruise [K]
RHO_0 = 1.18                # Unburned gas reference density [kg/m^3]
P_0 = RHO_0 * (U_0 ** 2)    # Dynamic pressure scale [Pa] (~ 2.71e4 Pa)
MU_0 = 1.85e-5              # Dynamic viscosity [Pa.s]
CP_0 = 1005.0               # Specific heat [J/kg.K]
R_SPECIFIC = 287.05         # Specific gas constant [J/kg.K]

# Dimensionless Numbers
RE = (RHO_0 * U_0 * L_0) / MU_0  # Reynolds number ~ 8.2e4
PR = 0.71                        # Molecular Prandtl number
SC = 0.65                        # Molecular Schmidt number
DA = 12.8                        # Damkohler number (combustion timescale ratio)
PR_T = 0.85                      # Turbulent Prandtl number
SC_T = 0.70                      # Turbulent Schmidt number

# Additional Physical Reference Scales for Multi-Physics Variables
P_REF = 1000.0              # Pressure scale [Pa]
OH_REF = 0.010              # Reference OH radical mole fraction [mol/mol]
NO_REF = 0.001              # Reference NO pollutant mole fraction [mol/mol] (1000 ppm)
Q_WALL_REF = 1.0e6          # Reference wall heat flux [W/m^2] (1 MW/m^2)

# ==============================================================================
# 4. MULTI-THROTTLE OPERATING CONDITIONS & PATHS
# ==============================================================================
ALL_DATA_DIRS = {
    0.55: r"D:\CFD\HYDROGEN NEW\RESULTS\SR_0.55",
    0.70: r"D:\CFD\HYDROGEN NEW\RESULTS\SR_0.70",
    0.895: r"D:\CFD\HYDROGEN NEW\RESULTS\SR_0.895 BASELINE",
    1.00: r"D:\CFD\HYDROGEN NEW\RESULTS\SR_1"
}

TRAIN_DATA_DIRS = ALL_DATA_DIRS  # In Option 1, data across all 4 throttles is pooled
HOLDOUT_DATA_DIR = r"D:\CFD\HYDROGEN NEW\RESULTS\SR_0.70"

# Train/Test Split (80% Train, 20% Test matching D:\CFD\PINN methodology)
TRAIN_TEST_SPLIT = 0.80

THROTTLE_CONDITIONS = {
    'idle': {
        'phi': 0.55,
        'p_th_kw': 85.7,
        'u_bulk': 65.4,
        't_in': 300.0,
        'role': 'train_test_80_20',
        'dir': ALL_DATA_DIRS[0.55]
    },
    'approach': {
        'phi': 0.70,
        'p_th_kw': 109.1,
        'u_bulk': 82.5,
        't_in': 300.0,
        'role': 'train_test_80_20',
        'dir': ALL_DATA_DIRS[0.70]
    },
    'cruise': {
        'phi': 0.895,
        'p_th_kw': 139.4,
        'u_bulk': 105.3,
        't_in': 300.0,
        'role': 'train_test_80_20',
        'dir': ALL_DATA_DIRS[0.895]
    },
    'takeoff': {
        'phi': 1.00,
        'p_th_kw': 155.7,
        'u_bulk': 118.2,
        't_in': 300.0,
        'role': 'train_test_80_20',
        'dir': ALL_DATA_DIRS[1.00]
    }
}

DATA_DIR = r"D:\CFD\HYDROGEN NEW\RESULTS"
EXP_DIR = r"D:\CFD\HYDROGEN NEW\EXPERIMENTAL_BENCHMARK_DATA"

# ==============================================================================
# 5. DATASET & COLLOCATION POINT BUDGET
# ==============================================================================
N_PDE_COLLOCATION = 80000   # Interior domain PDE collocation points
N_BC_COLLOCATION = 10000    # Boundary condition collocation points
N_RAR_CANDIDATES = 20000    # Candidate pool for adaptive refinement

BATCH_SIZE_PDE = 1024       # Mini-batch size for PDE collocation
BATCH_SIZE_DATA = 1024      # Mini-batch size for anchor data

# ==============================================================================
# 6. NEURAL NETWORK ARCHITECTURE
# ==============================================================================
FOURIER_SCALE_SIGMA = 2.5   # Gaussian bandwidth parameter for spectral bias elimination
FOURIER_NUM_FEATURES = 64   # 64 sine/cosine pairs -> 128 Fourier features (+ 1 for Phi = 129)
RESNET_NUM_BLOCKS = 6       # 6 residual blocks
RESNET_HIDDEN_DIM = 192     # 192 hidden neurons per block
ACTIVATION = "silu"         # Swish / SiLU activation for C^2 continuous derivatives

# ==============================================================================
# 7. LOSS FUNCTION WEIGHTS (Matched to D:\CFD\PINN 500:1 Data-to-PDE Ratio)
# ==============================================================================
WEIGHT_DATA = 50.0          # High-priority data anchor loss
WEIGHT_CONTINUITY = 0.1     # Continuity PDE regularizer
WEIGHT_MOMENTUM = 0.1       # Momentum Navier-Stokes regularizer
WEIGHT_SWIRL = 0.1          # Radial swirl equilibrium regularizer
WEIGHT_ENERGY = 0.05        # Thermal energy conservation regularizer
WEIGHT_SPECIES = 0.05       # Species conservation regularizer
WEIGHT_WALL = 10.0          # Wall heat flux loss weight
WEIGHT_PDROP = 5.0          # Macroscopic combustor pressure drop anchor

# ==============================================================================
# 8. TWO-STAGE OPTIMIZATION PARAMETERS
# ==============================================================================
ADAM_STEPS = 3000
ADAM_LR_INITIAL = 1e-3
ADAM_LR_FINAL = 1e-5

LBFGS_MAX_ITER = 500
LBFGS_LR = 0.5
LBFGS_HISTORY_SIZE = 50
LBFGS_LINE_SEARCH = "strong_wolfe"

RELOBRALO_TEMPERATURE = 0.1
RELOBRALO_ALPHA = 0.999
RELOBRALO_TAU = 0.1

RANDOM_SEED = 42
