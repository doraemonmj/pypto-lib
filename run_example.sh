#!/usr/bin/env bash
# Run a pypto-lib example with all required environment variables configured.
# Usage:
#   ./run_example.sh examples/beginner/hello_world.py
#   ./run_example.sh examples/beginner/hello_world.py -p a2a3sim
#   ./run_example.sh -l                              # list available examples

##### conda activate mj
# ── Environment variables ────────────────────────────────────────────────────
export TORCH_DEVICE_BACKEND_AUTOLOAD=0  # disable torch_npu auto-loading (version mismatch)
export PATH=/usr/local/bin/ptoas-bin:$PATH
export PTOAS_ROOT=/usr/local/bin/ptoas-bin
export PTO_ISA_ROOT="${PTO_ISA_ROOT:-/data/m00956180/runtime/pto-isa}"
export SIMPLER_ROOT="${SIMPLER_ROOT:-/data/m00956180/runtime/pypto-v2-ir/runtime}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"

# export SIMPLER_ROOT="${SIMPLER_ROOT:-$WORKSPACE_DIR/simpler-zhusy54}"
# export PTOAS_ROOT="${PTOAS_ROOT:-$WORKSPACE_DIR/ptoas-bin}"

# ── Run ──────────────────────────────────────────────────────────────────────
echo "=== Environment ==="
echo "  PTOAS_ROOT=$PTOAS_ROOT"
echo "  PTO_ISA_ROOT=$PTO_ISA_ROOT"
echo "  SIMPLER_ROOT=$SIMPLER_ROOT"
echo "  ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
EXAMPLE=examples/intermediate/wse-ffn.py
DEVICE=15
echo "=== Running: $EXAMPLE $@ (via task-submit --device $DEVICE) ==="
exec task-submit --device "$DEVICE" --run "python3 $EXAMPLE $*"