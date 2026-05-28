#!/bin/bash
set -euo pipefail

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

MACHINES=(
127.0.0.1
)
WORLD_SIZE=${#MACHINES[@]}
MASTER_ADDR="${MACHINES[0]}"
MASTER_PORT=29501
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
VENV_PATH="/opt/venv"
STEP=${1:?"Usage: $0 {high|low}"}

log "=== Multi-node training with $STEP ==="
mkdir -p logs/mova_distill_logs

RANK=0
for ip in "${MACHINES[@]}"; do
    LOG_FILE="logs/mova_distill_logs/node_${RANK}_${TIMESTAMP}.log"
    
    # 准备环境变量和命令
    CMD="
        source $VENV_PATH/bin/activate
        cd $(pwd)
        export NODE_RANK=$RANK
        export NNODES=$WORLD_SIZE
        export MASTER_ADDR=$MASTER_ADDR
        export MASTER_PORT=$MASTER_PORT
        bash scripts/distill/launch_distill.sh $STEP
    "
    
    if [ "$RANK" -eq 0 ]; then
        log "[Rank $RANK] Running locally on $ip, LOG_FILE=$LOG_FILE..."
        eval "$CMD" > "$LOG_FILE" 2>&1 &
    else
        log "[Rank $RANK] Launching on remote machine $ip, LOG_FILE=$LOG_FILE..."
        ssh -f "$ip" "$CMD" > "$LOG_FILE" 2>&1
    fi
    
    RANK=$((RANK + 1))
done

log "All $WORLD_SIZE nodes launched. Waiting for completion..."
wait
log "All tasks finished."