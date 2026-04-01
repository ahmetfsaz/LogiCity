#!/bin/bash
# Quick test for semantic GNA experiments.
# Sweeps a few gna_top_k values with both "semantic" and "semantic_random"
# selection modes to validate the simulation metrics pipeline.

# source /opt/conda/etc/profile.d/conda.sh
source /opt/anaconda3/etc/profile.d/conda.sh
conda init
conda activate logicity

# Configuration
CONFIG_FILE="config/tasks/sim/expert.yaml"
BACKUP_CONFIG="${CONFIG_FILE}.backup"
MAX_STEPS=100
NUM_TRIALS=3
LOG_DIR="log_sim"

GNA_TOP_K_VALUES=(0 1 3 5)
SELECTION_MODES=("semantic" "semantic_random" "semantic_lna" "semantic_lna_random" "semantic_lna_single" "semantic_lna_random_single")

# LNA modes: uplink transmits all FOV (k1 unused), sweep k2
LNA_K2_VALUES=(0 1 3 5)

RULE_FILE="${RULE_FILE:-config/rules/sim/expert/expert_rule_discriminative.yaml}"

RESULTS_DIR="semantic_quick_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

cp "$CONFIG_FILE" "$BACKUP_CONFIG"

echo "=========================================="
echo "Quick Test — Semantic GNA Experiments"
echo "=========================================="
echo "rule_file:            $RULE_FILE"
echo "Trials per config:    $NUM_TRIALS"
echo "gna_top_k values:     ${GNA_TOP_K_VALUES[*]}"
echo "LNA k2 values:        ${LNA_K2_VALUES[*]}"
echo "Selection modes:      ${SELECTION_MODES[*]}"
echo "Max steps:            $MAX_STEPS"
echo "Results:              $RESULTS_DIR"
echo "=========================================="
echo ""

update_config() {
    local top_k=$1
    local mode=$2
    local k1=${3:-3}
    local k2=${4:-3}
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

run_trials() {
    local mode=$1
    local label=$2  # display label for k
    shift 2
    # remaining args passed to update_config

    echo "=========================================="
    echo "Mode: $mode | k = $label"
    echo "=========================================="

    update_config "$@"

    declare -a sr_vals act_vals traj_vals
    declare -a sr_c_vals sr_i_vals sr_t_vals
    declare -a act_c_vals act_i_vals act_t_vals
    declare -a traj_total_vals traj_succ_vals traj_viol_vals traj_inc_vals

    for seed in $(seq 0 $((NUM_TRIALS - 1))); do
        EXPNAME="sem_qt_${mode}_${label}_s${seed}"
        LOG_FILE="${LOG_DIR}/${EXPNAME}.log"
        METRICS_FILE="${LOG_DIR}/${EXPNAME}_metrics.json"

        echo -n "  Trial $((seed + 1))/$NUM_TRIALS (seed=$seed)... "

        python3 main.py --config "$CONFIG_FILE" \
                --exp "$EXPNAME" \
                --max-steps $MAX_STEPS \
                --seed $seed \
                --log_dir "$LOG_DIR" > "$LOG_FILE" 2>&1

        metrics=$(extract_metrics "$METRICS_FILE")
        if [ -n "$metrics" ]; then
            read sr sr_c sr_i sr_t act act_c act_i act_t tr_rate tr_total tr_succ tr_viol tr_inc <<< "$metrics"
            sr_vals+=("$sr"); sr_c_vals+=("$sr_c"); sr_i_vals+=("$sr_i"); sr_t_vals+=("$sr_t")
            act_vals+=("$act"); act_c_vals+=("$act_c"); act_i_vals+=("$act_i"); act_t_vals+=("$act_t")
            traj_vals+=("$tr_rate"); traj_total_vals+=("$tr_total"); traj_succ_vals+=("$tr_succ")
            traj_viol_vals+=("$tr_viol"); traj_inc_vals+=("$tr_inc")
            echo "Done (SR=$sr [$sr_c/$sr_i/$sr_t]  Act=$act [$act_c/$act_i/$act_t]  Traj=$tr_rate [$tr_succ/$tr_total] viol=$tr_viol inc=$tr_inc)"
        else
            echo "Failed to extract metrics"
        fi
    done

    if [ ${#sr_vals[@]} -gt 0 ]; then
        sr_str=$(IFS=,; echo "${sr_vals[*]}")
        act_str=$(IFS=,; echo "${act_vals[*]}")
        traj_str=$(IFS=,; echo "${traj_vals[*]}")
        sr_c_str=$(IFS=,; echo "${sr_c_vals[*]}")
        sr_i_str=$(IFS=,; echo "${sr_i_vals[*]}")
        sr_t_str=$(IFS=,; echo "${sr_t_vals[*]}")
        act_c_str=$(IFS=,; echo "${act_c_vals[*]}")
        act_i_str=$(IFS=,; echo "${act_i_vals[*]}")
        act_t_str=$(IFS=,; echo "${act_t_vals[*]}")
        traj_total_str=$(IFS=,; echo "${traj_total_vals[*]}")
        traj_succ_str=$(IFS=,; echo "${traj_succ_vals[*]}")
        traj_viol_str=$(IFS=,; echo "${traj_viol_vals[*]}")
        traj_inc_str=$(IFS=,; echo "${traj_inc_vals[*]}")

        avgs=$(python3 -c "
sr=[${sr_str}]; act=[${act_str}]; tr=[${traj_str}]
sr_c=[${sr_c_str}]; sr_i=[${sr_i_str}]; sr_t=[${sr_t_str}]
act_c=[${act_c_str}]; act_i=[${act_i_str}]; act_t=[${act_t_str}]
tt=[${traj_total_str}]; ts=[${traj_succ_str}]; tv=[${traj_viol_str}]
ti=[${traj_inc_str}]
sr_m = sum(sr)/len(sr); act_m = sum(act)/len(act); tr_m = sum(tr)/len(tr)
sr_c_sum = int(sum(sr_c)); sr_i_sum = int(sum(sr_i)); sr_t_sum = int(sum(sr_t))
act_c_sum = int(sum(act_c)); act_i_sum = int(sum(act_i)); act_t_sum = int(sum(act_t))
tt_sum = int(sum(tt)); ts_sum = int(sum(ts)); tv_sum = int(sum(tv))
ti_sum = int(sum(ti))
print(f'{sr_m:.4f} {sr_c_sum}/{sr_i_sum}/{sr_t_sum} {act_m:.4f} {act_c_sum}/{act_i_sum}/{act_t_sum} {tr_m:.4f} {ts_sum}/{tt_sum} {tv_sum} {ti_sum}')
")
        read avg_sr sr_counts avg_act act_counts avg_traj comp viol inc <<< "$avgs"

        echo ""
        echo "  Averages over $NUM_TRIALS trials:"
        echo "    Subrule DSR:   $avg_sr  ($sr_counts)"
        echo "    Action DSR:    $avg_act  ($act_counts)"
        echo "    Trajectory SR: $avg_traj  (completed $comp | viol=$viol inc=$inc)"
        echo ""

        echo "$mode $label $avg_sr $sr_counts $avg_act $act_counts $avg_traj $comp $viol $inc" >> "${RESULTS_DIR}/summary.txt"
    else
        echo ""
        echo "  ERROR: No valid metrics collected"
        echo ""
    fi

    unset sr_vals act_vals traj_vals sr_c_vals sr_i_vals sr_t_vals act_c_vals act_i_vals act_t_vals traj_total_vals traj_succ_vals traj_viol_vals traj_inc_vals
}

for mode in "${SELECTION_MODES[@]}"; do
    if [[ "$mode" == semantic_lna* ]]; then
        for k2 in "${LNA_K2_VALUES[@]}"; do
            run_trials "$mode" "k2=$k2" 0 "$mode" 0 "$k2"
        done
    else
        for top_k in "${GNA_TOP_K_VALUES[@]}"; do
            run_trials "$mode" "$top_k" "$top_k" "$mode"
        done
    fi
done

# Restore config
cp "$BACKUP_CONFIG" "$CONFIG_FILE"
rm -f "$BACKUP_CONFIG"

# Final summary table
echo ""
echo "=========================================="
echo "QUICK TEST SUMMARY"
echo "=========================================="
printf "%-20s | %11s | %11s | %17s | %10s | %14s | %8s | %9s | %8s | %10s\n" \
       "Mode" "k" "Subrule DSR" "SR (C/I/Tot)" "Action DSR" "Act (C/I/Tot)" "Traj SR" "Completed" "Violated" "Incomplete"
printf "%-20s-|-%11s-|-%11s-|-%17s-|-%10s-|-%14s-|-%8s-|-%9s-|-%8s-|-%10s\n" \
       "--------------------" "-----------" "-----------" "-----------------" "----------" "--------------" "--------" "---------" "--------" "----------"

if [ -f "${RESULTS_DIR}/summary.txt" ]; then
    while read -r mode k sr sr_counts act act_counts traj comp viol inc; do
        printf "%-20s | %11s | %11s | %17s | %10s | %14s | %8s | %9s | %8s | %10s\n" \
               "$mode" "$k" "$sr" "$sr_counts" "$act" "$act_counts" "$traj" "$comp" "$viol" "$inc"
    done < "${RESULTS_DIR}/summary.txt"
fi

echo "=========================================="
echo ""
echo "Quick test completed! Results: $RESULTS_DIR"
echo "For full experiment: ./scripts/sim/run_semantic_experiments.sh"
echo ""
