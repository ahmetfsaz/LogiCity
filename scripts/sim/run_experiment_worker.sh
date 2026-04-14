#!/bin/bash
# Worker script: runs one experiment configuration through all 6 modes,
# k=0..5, 3 trials each. Writes results incrementally to RESULTS_FILE.
#
# Usage:
#   bash run_experiment_worker.sh CONFIG_FILE RULE_FILE LOG_DIR RESULTS_FILE LABEL
#
# CONFIG_FILE: per-experiment copy of expert.yaml (this script modifies it)
# RULE_FILE:   path to the rule yaml
# LOG_DIR:     directory for simulation logs
# RESULTS_FILE: append one line per (mode, k) after 3 trials complete
# LABEL:       human-readable experiment label (e.g., fov5_20c_8p_r70_original)

set -o pipefail

CONFIG_FILE="$1"
RULE_FILE="$2"
LOG_DIR="$3"
RESULTS_FILE="$4"
LABEL="$5"

MAX_STEPS=100
NUM_TRIALS=3
K_VALUES=(0 1 2 3 4 5)
SELECTION_MODES=("semantic" "semantic_random" "semantic_lna" "semantic_lna_random" "semantic_lna_single" "semantic_lna_random_single")

mkdir -p "$LOG_DIR"

WORKER_LOG="${LOG_DIR}/worker_${LABEL}.log"

log() {
    echo "[$(date '+%H:%M:%S')] [$LABEL] $*" | tee -a "$WORKER_LOG"
}

update_config() {
    local top_k=$1
    local mode=$2
    local k1=${3:-0}
    local k2=${4:-0}
    sed -i.tmp "s/gna_top_k:.*/gna_top_k: $top_k/" "$CONFIG_FILE"
    sed -i.tmp "s/gna_selection_mode:.*/gna_selection_mode: $mode/" "$CONFIG_FILE"
    sed -i.tmp "s/enable_sim_metrics:.*/enable_sim_metrics: true/" "$CONFIG_FILE"
    sed -i.tmp "s|rule_yaml_file:.*|rule_yaml_file: \"$RULE_FILE\"|" "$CONFIG_FILE"
    sed -i.tmp "s/gna_top_k1:.*/gna_top_k1: $k1/" "$CONFIG_FILE"
    sed -i.tmp "s/gna_top_k2:.*/gna_top_k2: $k2/" "$CONFIG_FILE"
    rm -f "${CONFIG_FILE}.tmp"
}

extract_metrics() {
    local json_file=$1
    python3 -c "
import json
with open('$json_file') as f:
    d = json.load(f)
sr  = d['decision_success_rate_subrule']
act = d['decision_success_rate_action']
t   = d['trajectory_success_rate']
print(f'{sr[\"average\"]:.6f} {sr[\"correct\"]} {sr[\"incorrect\"]} {sr[\"total_evaluated\"]} {act[\"average\"]:.6f} {act[\"correct\"]} {act[\"incorrect\"]} {act[\"total_evaluated\"]} {t[\"rate\"]:.6f} {t[\"total\"]} {t[\"successful\"]} {t[\"violated\"]} {t.get(\"incomplete\",0)}')
" 2>/dev/null
}

run_mode_k() {
    local mode=$1
    local k_label=$2
    shift 2

    update_config "$@"

    declare -a sr_vals act_vals traj_vals
    declare -a sr_c_vals sr_i_vals sr_t_vals
    declare -a act_c_vals act_i_vals act_t_vals
    declare -a traj_total_vals traj_succ_vals traj_viol_vals traj_inc_vals

    for seed in $(seq 0 $((NUM_TRIALS - 1))); do
        EXPNAME="${LABEL}_${mode}_${k_label}_s${seed}"
        SIM_LOG="${LOG_DIR}/${EXPNAME}.log"
        METRICS_FILE="${LOG_DIR}/${EXPNAME}_metrics.json"

        log "  Trial $((seed + 1))/$NUM_TRIALS mode=$mode k=$k_label seed=$seed"

        python3 main.py --config "$CONFIG_FILE" \
                --exp "$EXPNAME" \
                --max-steps $MAX_STEPS \
                --seed $seed \
                --log_dir "$LOG_DIR" > "$SIM_LOG" 2>&1

        metrics=$(extract_metrics "$METRICS_FILE")
        if [ -n "$metrics" ]; then
            read sr sr_c sr_i sr_t act act_c act_i act_t tr_rate tr_total tr_succ tr_viol tr_inc <<< "$metrics"
            sr_vals+=("$sr"); sr_c_vals+=("$sr_c"); sr_i_vals+=("$sr_i"); sr_t_vals+=("$sr_t")
            act_vals+=("$act"); act_c_vals+=("$act_c"); act_i_vals+=("$act_i"); act_t_vals+=("$act_t")
            traj_vals+=("$tr_rate"); traj_total_vals+=("$tr_total"); traj_succ_vals+=("$tr_succ")
            traj_viol_vals+=("$tr_viol"); traj_inc_vals+=("$tr_inc")
        else
            log "  WARNING: Failed to extract metrics for $EXPNAME"
        fi
    done

    if [ ${#sr_vals[@]} -gt 0 ]; then
        local sr_str=$(IFS=,; echo "${sr_vals[*]}")
        local act_str=$(IFS=,; echo "${act_vals[*]}")
        local traj_str=$(IFS=,; echo "${traj_vals[*]}")
        local sr_c_str=$(IFS=,; echo "${sr_c_vals[*]}")
        local sr_i_str=$(IFS=,; echo "${sr_i_vals[*]}")
        local sr_t_str=$(IFS=,; echo "${sr_t_vals[*]}")
        local act_c_str=$(IFS=,; echo "${act_c_vals[*]}")
        local act_i_str=$(IFS=,; echo "${act_i_vals[*]}")
        local act_t_str=$(IFS=,; echo "${act_t_vals[*]}")
        local traj_total_str=$(IFS=,; echo "${traj_total_vals[*]}")
        local traj_succ_str=$(IFS=,; echo "${traj_succ_vals[*]}")
        local traj_viol_str=$(IFS=,; echo "${traj_viol_vals[*]}")
        local traj_inc_str=$(IFS=,; echo "${traj_inc_vals[*]}")

        local stats
        stats=$(python3 -c "
import math
sr=[${sr_str}]; act=[${act_str}]; tr=[${traj_str}]
sr_c=[${sr_c_str}]; sr_i=[${sr_i_str}]; sr_t=[${sr_t_str}]
act_c=[${act_c_str}]; act_i=[${act_i_str}]; act_t=[${act_t_str}]
tt=[${traj_total_str}]; ts=[${traj_succ_str}]; tv=[${traj_viol_str}]
ti=[${traj_inc_str}]
def mean_std(vals):
    n = len(vals); m = sum(vals) / n
    s = math.sqrt(sum((v-m)**2 for v in vals)/(n-1)) if n > 1 else 0.0
    return m, s
sr_m,sr_s = mean_std(sr); act_m,act_s = mean_std(act); tr_m,tr_s = mean_std(tr)
print(f'{sr_m:.6f} {sr_s:.6f} {int(sum(sr_c))} {int(sum(sr_i))} {int(sum(sr_t))} {act_m:.6f} {act_s:.6f} {int(sum(act_c))} {int(sum(act_i))} {int(sum(act_t))} {tr_m:.6f} {tr_s:.6f} {int(sum(tt))} {int(sum(ts))} {int(sum(tv))} {int(sum(ti))}')
")
        read sr_m sr_s sr_c_sum sr_i_sum sr_t_sum act_m act_s act_c_sum act_i_sum act_t_sum tr_m tr_s tt_sum ts_sum tv_sum ti_sum <<< "$stats"

        echo "$LABEL $mode $k_label $sr_m $sr_s $sr_c_sum $sr_i_sum $sr_t_sum $act_m $act_s $act_c_sum $act_i_sum $act_t_sum $tr_m $tr_s $tt_sum $ts_sum $tv_sum $ti_sum" >> "$RESULTS_FILE"

        log "  Done mode=$mode k=$k_label  SR=${sr_m}+/-${sr_s}  Act=${act_m}+/-${act_s}  Traj=${tr_m}+/-${tr_s}"
    else
        echo "$LABEL $mode $k_label FAIL 0 0 0 0 FAIL 0 0 0 0 FAIL 0 0 0 0 0" >> "$RESULTS_FILE"
        log "  FAILED mode=$mode k=$k_label"
    fi

    unset sr_vals act_vals traj_vals sr_c_vals sr_i_vals sr_t_vals act_c_vals act_i_vals act_t_vals traj_total_vals traj_succ_vals traj_viol_vals traj_inc_vals
}

log "Starting experiment: $LABEL"
log "  Config: $CONFIG_FILE"
log "  Rules:  $RULE_FILE"

for mode in "${SELECTION_MODES[@]}"; do
    log "Mode: $mode"
    if [[ "$mode" == semantic_lna* ]]; then
        for k in "${K_VALUES[@]}"; do
            run_mode_k "$mode" "k2=$k" 0 "$mode" 0 "$k"
        done
    else
        for k in "${K_VALUES[@]}"; do
            run_mode_k "$mode" "$k" "$k" "$mode"
        done
    fi
done

log "Experiment $LABEL completed."
