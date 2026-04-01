#!/bin/bash
# Verification script: run gna_top_k=0,1,2,3 with semantic and semantic_random
# to confirm metrics produce varying results across configurations.

source /opt/anaconda3/etc/profile.d/conda.sh
conda init
conda activate logicity

CONFIG_FILE="config/tasks/sim/expert.yaml"
BACKUP_CONFIG="${CONFIG_FILE}.backup"
MAX_STEPS=${MAX_STEPS:-100}
LOG_DIR="log_sim"
SEED=0

GNA_TOP_K_VALUES=(0 1 2 3)
SELECTION_MODES=("semantic_lna" "semantic_lna_random" "semantic_lna_single" "semantic_lna_random_single")

# LNA modes: uplink transmits all FOV (k1 unused), sweep k2
LNA_K2_VALUES=(0 1 2 3 4 5)

cp "$CONFIG_FILE" "$BACKUP_CONFIG"

RULE_FILE="${RULE_FILE:-config/rules/sim/expert/expert_rule_discriminative.yaml}"

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

declare -a ALL_RESULTS

echo "=========================================="
echo " Metrics Verification Sweep"
echo "=========================================="
echo " rule_file:    $RULE_FILE"
echo " top_k values: ${GNA_TOP_K_VALUES[*]}"
echo " LNA k2:       ${LNA_K2_VALUES[*]}"
echo " modes:        ${SELECTION_MODES[*]}"
echo " steps=$MAX_STEPS  seed=$SEED"
echo "=========================================="
echo ""

for mode in "${SELECTION_MODES[@]}"; do
    if [[ "$mode" == semantic_lna* ]]; then
        # LNA modes: uplink transmits all FOV (k1 unused), sweep k2
        for k2 in "${LNA_K2_VALUES[@]}"; do
            update_config 0 "$mode" 0 "$k2"

            EXPNAME="verify_${mode}_k2${k2}_s${SEED}"
            METRICS_FILE="${LOG_DIR}/${EXPNAME}_metrics.json"

            echo -n "Running mode=$mode k2=$k2 ... "

            python3 main.py --config "$CONFIG_FILE" \
                    --exp "$EXPNAME" \
                    --max-steps $MAX_STEPS \
                    --seed $SEED \
                    --log_dir "$LOG_DIR" > /dev/null 2>&1

            metrics=$(extract_metrics "$METRICS_FILE")
            if [ -n "$metrics" ]; then
                read sr sr_c sr_i sr_t act act_c act_i act_t tr_rate tr_total tr_succ tr_viol tr_inc <<< "$metrics"
                echo "OK  SR=$sr [$sr_c/$sr_i/$sr_t]  Act=$act [$act_c/$act_i/$act_t]  Traj=$tr_rate [$tr_succ/$tr_total] viol=$tr_viol inc=$tr_inc"
                ALL_RESULTS+=("$mode k2=$k2 $sr $sr_c/$sr_i/$sr_t $act $act_c/$act_i/$act_t $tr_rate $tr_succ/$tr_total $tr_viol $tr_inc")
            else
                echo "FAILED"
                ALL_RESULTS+=("$mode k2=$k2 FAIL - FAIL - FAIL - - -")
            fi
        done
    else
        # Standard modes: sweep top_k
        for top_k in "${GNA_TOP_K_VALUES[@]}"; do
            update_config "$top_k" "$mode"

            EXPNAME="verify_${mode}_k${top_k}_s${SEED}"
            METRICS_FILE="${LOG_DIR}/${EXPNAME}_metrics.json"

            echo -n "Running mode=$mode top_k=$top_k ... "

            python3 main.py --config "$CONFIG_FILE" \
                    --exp "$EXPNAME" \
                    --max-steps $MAX_STEPS \
                    --seed $SEED \
                    --log_dir "$LOG_DIR" > /dev/null 2>&1

            metrics=$(extract_metrics "$METRICS_FILE")
            if [ -n "$metrics" ]; then
                read sr sr_c sr_i sr_t act act_c act_i act_t tr_rate tr_total tr_succ tr_viol tr_inc <<< "$metrics"
                echo "OK  SR=$sr [$sr_c/$sr_i/$sr_t]  Act=$act [$act_c/$act_i/$act_t]  Traj=$tr_rate [$tr_succ/$tr_total] viol=$tr_viol inc=$tr_inc"
                ALL_RESULTS+=("$mode $top_k $sr $sr_c/$sr_i/$sr_t $act $act_c/$act_i/$act_t $tr_rate $tr_succ/$tr_total $tr_viol $tr_inc")
            else
                echo "FAILED"
                ALL_RESULTS+=("$mode $top_k FAIL - FAIL - FAIL - - -")
            fi
        done
    fi
done

cp "$BACKUP_CONFIG" "$CONFIG_FILE"
rm -f "$BACKUP_CONFIG"

echo ""
echo "=========================================="
echo " RESULTS"
echo "=========================================="
printf "%-20s | %11s | %11s | %17s | %10s | %14s | %8s | %9s | %8s | %10s\n" \
       "Mode" "k" "Subrule DSR" "SR (C/I/Tot)" "Action DSR" "Act (C/I/Tot)" "Traj SR" "Completed" "Violated" "Incomplete"
printf "%-20s-|-%11s-|-%11s-|-%17s-|-%10s-|-%14s-|-%8s-|-%9s-|-%8s-|-%10s\n" \
       "--------------------" "-----------" "-----------" "-----------------" "----------" "--------------" "--------" "---------" "--------" "----------"
for row in "${ALL_RESULTS[@]}"; do
    read mode top_k sr sr_counts act act_counts tr comp viol inc <<< "$row"
    printf "%-20s | %11s | %11s | %17s | %10s | %14s | %8s | %9s | %8s | %10s\n" \
           "$mode" "$top_k" "$sr" "$sr_counts" "$act" "$act_counts" "$tr" "$comp" "$viol" "$inc"
done
echo "=========================================="
echo ""
echo "Verification complete."
