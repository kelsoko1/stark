# ─────────────────────────────────────────────────────────────────
# Q-BitNet + LingBot-World v3 — Configuration
# Edit this file to change training settings
# ─────────────────────────────────────────────────────────────────

# Model
BASE_MODEL_ID   = "microsoft/bitnet-b1.58-2B-4T-bf16"
MODEL_LOCAL_DIR = "./models/bitnet-2b4t-bf16"

# Training
BATCH_SIZE      = 1        # reduced for 16GB RAM
GRAD_ACCUM      = 4        # effective batch = 4
MAX_SEQ_LEN     = 128      # reduced from 512 — saves ~75% activation memory
LEARNING_RATE   = 2e-4     # higher LR since only training small additions
INNER_STEPS     = 500
OUTER_LR        = 0.7
MAX_EPOCHS      = 10
SAVE_EVERY      = 1
LOG_EVERY       = 1        # log every step so we see progress immediately

# Freeze base model — only train Q-BitNet additions (saves ~10 GB RAM)
FREEZE_BASE_MODEL = True   # IMPORTANT: set False only if you have 32+ GB RAM

# Q-BitNet modifications — toggle each on/off
USE_SUPERPOSITION   = True   # differentiable ternary weights
USE_INTERFERENCE    = False  # quantum interference attention (needs more RAM)
USE_ENTANGLEMENT    = True   # entangled head pairs — drives Phi
USE_PHASE_STREAM    = True   # phase-encoded residual stream
USE_HADAMARD        = False  # Walsh-Hadamard mixing (disabled — FWHT shape bug)
USE_TUNNELING       = True   # quantum tunneling optimizer
ENTANGLE_TOP_K      = 6      # how many top layers to entangle

# LingBot brain modules — toggle each on/off
USE_INTEROCEPTION   = True
USE_HIPPOCAMPUS     = True
USE_AMYGDALA        = True
USE_FREE_ENERGY     = True
HIPPOCAMPUS_BUFFER  = 2048   # number of episodes to store

# Optimizer (QuantumTunnelingMuon)
TUNNEL_P0       = 0.02       # initial tunneling probability
TUNNEL_DECAY    = 5000       # steps to decay tunneling to near zero
TUNNEL_SCALE    = 8.0        # step size during tunnel event
MOMENTUM        = 0.95

# Data — defaults to TinyStories (free, downloads automatically)
DATASET_NAME    = "roneneldan/TinyStories"
DATASET_SPLIT   = "train"

# Checkpoints
CHECKPOINT_DIR  = "./checkpoints"

# WebSocket server (dashboard communication)
WS_HOST         = "localhost"
WS_PORT         = 8765
