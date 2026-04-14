#!/bin/bash
# Master orchestrator: runs 18 experiment configurations across 3 FOV groups.
# Each experiment runs all 6 modes x k=0..5 x 3 trials = 108 simulation runs.
#
# FOV groups run sequentially (AGENT_FOV is a global constant).
# Within each group, up to MAX_PARALLEL experiments run concurrently.

set -o pipefail

# source /opt/conda/etc/profile.d/conda.sh
source /opt/anaconda3/etc/profile.d/conda.sh
conda init
conda activate logicity

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_ROOT="full_experiments_${TIMESTAMP}"
CONFIGS_DIR="${RESULTS_ROOT}/configs"
AGENTS_DIR="${RESULTS_ROOT}/agents"
LOGS_DIR="${RESULTS_ROOT}/logs"

mkdir -p "$CONFIGS_DIR" "$AGENTS_DIR" "$LOGS_DIR"

CONFIG_PY="logicity/core/config.py"
ORIG_EXPERT="config/tasks/sim/expert.yaml"
MAX_PARALLEL=5

ORIG_FOV=$(sed -n 's/^AGENT_FOV *= *\([0-9]*\).*/\1/p' "$CONFIG_PY")
echo "Original AGENT_FOV = $ORIG_FOV (will be restored at the end)"

# Rule file paths (relative to project root)
RULE_ORIGINAL="config/rules/sim/expert/expert_rule.yaml"
RULE_EXTENDED="config/rules/sim/expert/expert_rule_extended.yaml"
RULE_SPATIAL="config/rules/sim/expert/expert_rule_spatial.yaml"
RULE_DISCRIMINATIVE="config/rules/sim/expert/expert_rule_discriminative.yaml"

# ── Generate agent YAML variants ──
gen_agents() {
    local cars=$1 peds=$2 tag=$3
    local out="${AGENTS_DIR}/agents_${tag}.yaml"
    python3 "$SCRIPT_DIR/generate_agents.py" --cars "$cars" --peds "$peds" --output "$out"
    echo "$out"
}

AGENTS_10c_8p=$(gen_agents 10 8 "10c_8p")
AGENTS_20c_4p=$(gen_agents 20 4 "20c_4p")
AGENTS_20c_8p=$(gen_agents 20 8 "20c_8p")
AGENTS_20c_12p=$(gen_agents 20 12 "20c_12p")
AGENTS_30c_8p=$(gen_agents 30 8 "30c_8p")

# ── Create per-experiment config files ──
make_config() {
    local tag=$1 agent_yaml=$2 rule_yaml=$3 region=$4
    local cfg="${CONFIGS_DIR}/expert_${tag}.yaml"
    cp "$ORIG_EXPERT" "$cfg"
    sed -i.tmp "s|agent_yaml_file:.*|agent_yaml_file: \"$agent_yaml\"|" "$cfg"
    sed -i.tmp "s|rule_yaml_file:.*|rule_yaml_file: \"$rule_yaml\"|" "$cfg"
    sed -i.tmp "s/agent_region:.*/agent_region: $region/" "$cfg"
    sed -i.tmp "s/enable_sim_metrics:.*/enable_sim_metrics: true/" "$cfg"
    sed -i.tmp "s/enable_gna:.*/enable_gna: true/" "$cfg"
    rm -f "${cfg}.tmp"
    echo "$cfg"
}

# Declare experiment arrays: LABEL CONFIG RULE FOV
declare -a EXP_LABELS EXP_CONFIGS EXP_RULES EXP_FOVS

add_experiment() {
    local fov=$1 label=$2 config=$3 rule=$4
    EXP_LABELS+=("$label")
    EXP_CONFIGS+=("$config")
    EXP_RULES+=("$rule")
    EXP_FOVS+=("$fov")
}

# ── FOV=5 experiments — ALL COMPLETED in previous run ──
# add_experiment 5 "fov5_20c8p_r70_original"       "$(make_config "fov5_20c8p_r70_original"       "$AGENTS_20c_8p" "$RULE_ORIGINAL"       70)" "$RULE_ORIGINAL"
# add_experiment 5 "fov5_20c8p_r70_extended"        "$(make_config "fov5_20c8p_r70_extended"        "$AGENTS_20c_8p" "$RULE_EXTENDED"       70)" "$RULE_EXTENDED"
# add_experiment 5 "fov5_20c8p_r70_spatial"         "$(make_config "fov5_20c8p_r70_spatial"         "$AGENTS_20c_8p" "$RULE_SPATIAL"        70)" "$RULE_SPATIAL"
# add_experiment 5 "fov5_20c8p_r70_discriminative"  "$(make_config "fov5_20c8p_r70_discriminative"  "$AGENTS_20c_8p" "$RULE_DISCRIMINATIVE" 70)" "$RULE_DISCRIMINATIVE"

# ── FOV=7 experiments — all completed except fov7_30c8p (partial, very slow) ──
# add_experiment 7 "fov7_20c8p_r70_original"        "$(make_config "fov7_20c8p_r70_original"        "$AGENTS_20c_8p"  "$RULE_ORIGINAL"       70)" "$RULE_ORIGINAL"
# add_experiment 7 "fov7_20c8p_r70_extended"         "$(make_config "fov7_20c8p_r70_extended"         "$AGENTS_20c_8p"  "$RULE_EXTENDED"       70)" "$RULE_EXTENDED"
# add_experiment 7 "fov7_20c8p_r70_spatial"          "$(make_config "fov7_20c8p_r70_spatial"          "$AGENTS_20c_8p"  "$RULE_SPATIAL"        70)" "$RULE_SPATIAL"
# add_experiment 7 "fov7_20c8p_r70_discriminative"   "$(make_config "fov7_20c8p_r70_discriminative"   "$AGENTS_20c_8p"  "$RULE_DISCRIMINATIVE" 70)" "$RULE_DISCRIMINATIVE"
# add_experiment 7 "fov7_20c4p_r70_discriminative"   "$(make_config "fov7_20c4p_r70_discriminative"   "$AGENTS_20c_4p"  "$RULE_DISCRIMINATIVE" 70)" "$RULE_DISCRIMINATIVE"
# add_experiment 7 "fov7_20c12p_r70_discriminative"  "$(make_config "fov7_20c12p_r70_discriminative"  "$AGENTS_20c_12p" "$RULE_DISCRIMINATIVE" 70)" "$RULE_DISCRIMINATIVE"
# add_experiment 7 "fov7_10c8p_r70_discriminative"   "$(make_config "fov7_10c8p_r70_discriminative"   "$AGENTS_10c_8p"  "$RULE_DISCRIMINATIVE" 70)" "$RULE_DISCRIMINATIVE"
# add_experiment 7 "fov7_30c8p_r70_discriminative"   "$(make_config "fov7_30c8p_r70_discriminative"   "$AGENTS_30c_8p"  "$RULE_DISCRIMINATIVE" 70)" "$RULE_DISCRIMINATIVE"
# add_experiment 7 "fov7_20c8p_r50_discriminative"   "$(make_config "fov7_20c8p_r50_discriminative"   "$AGENTS_20c_8p"  "$RULE_DISCRIMINATIVE" 50)" "$RULE_DISCRIMINATIVE"
# add_experiment 7 "fov7_20c8p_r120_discriminative"  "$(make_config "fov7_20c8p_r120_discriminative"  "$AGENTS_20c_8p"  "$RULE_DISCRIMINATIVE" 120)" "$RULE_DISCRIMINATIVE"

# ── FOV=10 experiments (4) ──
add_experiment 10 "fov10_20c8p_r70_original"       "$(make_config "fov10_20c8p_r70_original"       "$AGENTS_20c_8p" "$RULE_ORIGINAL"       70)" "$RULE_ORIGINAL"
add_experiment 10 "fov10_20c8p_r70_extended"        "$(make_config "fov10_20c8p_r70_extended"        "$AGENTS_20c_8p" "$RULE_EXTENDED"       70)" "$RULE_EXTENDED"
add_experiment 10 "fov10_20c8p_r70_spatial"         "$(make_config "fov10_20c8p_r70_spatial"         "$AGENTS_20c_8p" "$RULE_SPATIAL"        70)" "$RULE_SPATIAL"
add_experiment 10 "fov10_20c8p_r70_discriminative"  "$(make_config "fov10_20c8p_r70_discriminative"  "$AGENTS_20c_8p" "$RULE_DISCRIMINATIVE" 70)" "$RULE_DISCRIMINATIVE"

N_EXPERIMENTS=${#EXP_LABELS[@]}
echo ""
echo "=============================================="
echo "  Full Experiment Suite — $N_EXPERIMENTS experiments"
echo "=============================================="
echo "  Max parallel:     $MAX_PARALLEL"
echo "  Results dir:      $RESULTS_ROOT"
echo "  Agent variants:   $AGENTS_DIR"
echo "  Config copies:    $CONFIGS_DIR"
echo "  Log directory:    $LOGS_DIR"
echo "=============================================="
echo ""

# ── Job management ──
declare -a ACTIVE_PIDS=()
declare -a ACTIVE_LABELS=()

wait_for_slot() {
    while [ ${#ACTIVE_PIDS[@]} -ge $MAX_PARALLEL ]; do
        local new_pids=() new_labels=()
        for _idx in "${!ACTIVE_PIDS[@]}"; do
            if kill -0 "${ACTIVE_PIDS[$_idx]}" 2>/dev/null; then
                new_pids+=("${ACTIVE_PIDS[$_idx]}")
                new_labels+=("${ACTIVE_LABELS[$_idx]}")
            else
                wait "${ACTIVE_PIDS[$_idx]}" 2>/dev/null || true
                echo "[$(date '+%H:%M:%S')] Finished: ${ACTIVE_LABELS[$_idx]} (PID ${ACTIVE_PIDS[$_idx]})"
            fi
        done
        ACTIVE_PIDS=("${new_pids[@]+"${new_pids[@]}"}")
        ACTIVE_LABELS=("${new_labels[@]+"${new_labels[@]}"}")
        if [ ${#ACTIVE_PIDS[@]} -ge $MAX_PARALLEL ]; then
            sleep 10
        fi
    done
}

wait_all() {
    for _idx in "${!ACTIVE_PIDS[@]}"; do
        wait "${ACTIVE_PIDS[$_idx]}" 2>/dev/null || true
        echo "[$(date '+%H:%M:%S')] Finished: ${ACTIVE_LABELS[$_idx]} (PID ${ACTIVE_PIDS[$_idx]})"
    done
    ACTIVE_PIDS=()
    ACTIVE_LABELS=()
}

set_fov() {
    local fov=$1
    sed -i.tmp "s/AGENT_FOV *= *[0-9]*/AGENT_FOV = $fov/" "$CONFIG_PY"
    rm -f "${CONFIG_PY}.tmp"
    echo "[$(date '+%H:%M:%S')] Set AGENT_FOV = $fov"
}

# ── Run experiments grouped by FOV ──
for target_fov in 5 7 10; do
    echo ""
    echo "######################################################"
    echo "#  FOV GROUP: AGENT_FOV = $target_fov"
    echo "######################################################"
    echo ""

    set_fov "$target_fov"

    for i in $(seq 0 $((N_EXPERIMENTS - 1))); do
        if [ "${EXP_FOVS[$i]}" != "$target_fov" ]; then
            continue
        fi

        wait_for_slot

        local_label="${EXP_LABELS[$i]}"
        local_config="${EXP_CONFIGS[$i]}"
        local_rule="${EXP_RULES[$i]}"
        local_logdir="${LOGS_DIR}/${local_label}"
        local_results="${RESULTS_ROOT}/${local_label}_results.txt"

        mkdir -p "$local_logdir"

        echo "[$(date '+%H:%M:%S')] Launching: $local_label"

        bash "$SCRIPT_DIR/run_experiment_worker.sh" \
            "$local_config" "$local_rule" "$local_logdir" "$local_results" "$local_label" &

        ACTIVE_PIDS+=($!)
        ACTIVE_LABELS+=("$local_label")
    done

    echo "[$(date '+%H:%M:%S')] Waiting for FOV=$target_fov group to finish..."
    wait_all
    echo "[$(date '+%H:%M:%S')] FOV=$target_fov group complete."
done

# ── Restore original FOV ──
set_fov "$ORIG_FOV"
echo ""
echo "[$(date '+%H:%M:%S')] Restored AGENT_FOV = $ORIG_FOV"

# ── Aggregate results ──
echo ""
echo "######################################################"
echo "#  AGGREGATING RESULTS"
echo "######################################################"
echo ""

MASTER_CSV="${RESULTS_ROOT}/summary.csv"
MASTER_TXT="${RESULTS_ROOT}/summary.txt"

{
    echo "label,mode,k,sr_mean,sr_std,sr_correct,sr_incorrect,sr_total,act_mean,act_std,act_correct,act_incorrect,act_total,traj_mean,traj_std,traj_total,traj_success,traj_violated,traj_incomplete"
    for resfile in "${RESULTS_ROOT}"/*_results.txt; do
        [ -f "$resfile" ] || continue
        while read -r label mode k sr_m sr_s sr_c sr_i sr_t act_m act_s act_c act_i act_t tr_m tr_s tt ts tv ti; do
            echo "$label,$mode,$k,$sr_m,$sr_s,$sr_c,$sr_i,$sr_t,$act_m,$act_s,$act_c,$act_i,$act_t,$tr_m,$tr_s,$tt,$ts,$tv,$ti"
        done < "$resfile"
    done
} > "$MASTER_CSV"

# Generate formatted summary
{
    echo "=============================================================="
    echo "  FULL EXPERIMENT SUMMARY — $(date)"
    echo "=============================================================="
    echo ""

    current_label=""
    for resfile in "${RESULTS_ROOT}"/*_results.txt; do
        [ -f "$resfile" ] || continue
        exp_label=$(basename "$resfile" _results.txt)
        echo "────────────────────────────────────────────────────────────────"
        echo "Experiment: $exp_label"
        echo "────────────────────────────────────────────────────────────────"
        printf "%-22s %-8s | %-18s | %-18s | %-18s\n" \
               "Mode" "k" "Subrule DSR" "Action DSR" "Trajectory SR"
        printf "%-22s-%-8s-|-%-18s-|-%-18s-|-%-18s\n" \
               "----------------------" "--------" "------------------" "------------------" "------------------"

        while read -r label mode k sr_m sr_s sr_c sr_i sr_t act_m act_s act_c act_i act_t tr_m tr_s tt ts tv ti; do
            printf "%-22s %-8s | %7s ± %-8s | %7s ± %-8s | %7s ± %-8s\n" \
                   "$mode" "$k" "$sr_m" "$sr_s" "$act_m" "$act_s" "$tr_m" "$tr_s"
        done < "$resfile"
        echo ""
    done
} > "$MASTER_TXT"

cat "$MASTER_TXT"

echo ""
echo "=============================================="
echo "  RESULTS SAVED"
echo "=============================================="
echo "  Master CSV:   $MASTER_CSV"
echo "  Summary:      $MASTER_TXT"
echo "  Per-experiment: ${RESULTS_ROOT}/<label>_results.txt"
echo "  Logs:         $LOGS_DIR/"
echo "=============================================="
echo ""
echo "Full experiment suite completed!"
