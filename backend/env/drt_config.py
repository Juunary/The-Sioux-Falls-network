# ============================================================
# DRT environment constants
# Ported from KW_DRT/app/config.py (TF/param_grid removed)
# ============================================================

# Fleet
MAX_NUM_VEHICLES: int = 2
VEH_CAPACITY: int = 5

# Request slots
MAX_NUM_REQUEST: int = 8
POSSIBLE_ACTION: int = MAX_NUM_REQUEST + 1   # 0..7 = request slots, 8 = REJECT

# Time limits
MAX_WAIT_TIME: int = 10       # ticks before request is cancelled
MAX_INVEHICLE_TIME: int = 10  # ticks (used in reward normalisation)

# Network (confirmed from KW_DRT/data/od_matrix.csv header: 24 columns)
NUM_NODES: int = 24

# Observation space flat vector length (must match drt_observation.py)
# Layout:
#   [0:24]   self_curr_node one-hot
#   [24:48]  fleet_curr_node distribution
#   [48]     self_capacity_ratio
#   [49:52]  self_status one-hot (IDLE/PICKUP/DROPOFF)
#   [52]     time_normalized
#   [53:157] 8 request slots × 13 dims each
OBS_DIM_DRT: int = 157

# Maximum episode time (used for time normalisation)
MAX_EPISODE_TIME: int = 200
