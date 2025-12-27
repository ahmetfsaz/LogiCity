#!/bin/bash
# Experiment script to test different agent_region values with multiple random seeds
# Tests agent_region from 70 to 200 with 20 trials each

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

# Agent region values to test (from 70 to 200, step by 10)
AGENT_REGIONS=(200 190 170 100 110 120 130 140 150 160 170 180 190 200)

# Create results directory
RESULTS_DIR="experiment_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Backup original config
cp "$CONFIG_FILE" "$BACKUP_CONFIG"

echo "=========================================="
echo "Agent Region Experiment - Subrule Analysis"
echo "=========================================="
echo "Number of trials per agent_region: $NUM_TRIALS"
echo "Agent regions to test: ${AGENT_REGIONS[@]}"
echo "Max steps per trial: $MAX_STEPS"
echo "Results directory: $RESULTS_DIR"
echo "=========================================="
echo ""

# Function to update agent_region in config file
update_agent_region() {
    local region=$1
    # Use sed to replace the agent_region line
    sed -i.tmp "s/agent_region:.*/agent_region: $region/" "$CONFIG_FILE"
    rm -f "${CONFIG_FILE}.tmp"
}

# Function to extract metrics from log file
extract_metrics() {
    local log_file=$1
    # Extract the last occurrence of "Per-Agent Averages" line
    grep "Per-Agent Averages" "$log_file" | tail -1 | \
        awk -F'Fully: |, Partially: |, Unobserved: | \\(Agents' '{print $2, $3, $4}'
}

# Main experiment loop
for region in "${AGENT_REGIONS[@]}"; do
    echo "=========================================="
    echo "Testing agent_region = $region"
    echo "=========================================="
    
    # Update config file
    update_agent_region "$region"
    echo "Updated $CONFIG_FILE with agent_region: $region"
    
    # Arrays to store metrics across trials
    declare -a fully_values
    declare -a partially_values
    declare -a unobserved_values
    
    # Run trials
    for seed in $(seq 0 $((NUM_TRIALS - 1))); do
        EXPNAME="expert_region${region}_seed${seed}"
        LOG_FILE="${LOG_DIR}/${EXPNAME}.log"
        
        echo -n "  Trial $((seed + 1))/$NUM_TRIALS (seed=$seed)... "
        
        # Run experiment and redirect output to log file
        python3 main.py --config "$CONFIG_FILE" \
                --exp "$EXPNAME" \
                --max-steps $MAX_STEPS \
                --seed $seed \
                --log_dir "$LOG_DIR" > "$LOG_FILE" 2>&1
        
        # Extract metrics
        metrics=$(extract_metrics "$LOG_FILE")
        
        if [ -n "$metrics" ]; then
            read fully partially unobserved <<< "$metrics"
            fully_values+=("$fully")
            partially_values+=("$partially")
            unobserved_values+=("$unobserved")
            echo "Done (Fully: $fully, Partially: $partially, Unobserved: $unobserved)"
        else
            echo "Failed to extract metrics"
        fi
    done
    
    # Calculate averages
    if [ ${#fully_values[@]} -gt 0 ]; then
        echo ""
        echo "  Computing averages across $NUM_TRIALS trials..."
        
        # Use Python to calculate averages
        # Convert bash arrays to comma-separated values for Python
        fully_str=$(IFS=,; echo "${fully_values[*]}")
        partially_str=$(IFS=,; echo "${partially_values[*]}")
        unobserved_str=$(IFS=,; echo "${unobserved_values[*]}")
        
        avg_metrics=$(python3 -c "
import sys
fully = [${fully_str}]
partially = [${partially_str}]
unobserved = [${unobserved_str}]

avg_fully = sum(fully) / len(fully) if fully else 0
avg_partially = sum(partially) / len(partially) if partially else 0
avg_unobserved = sum(unobserved) / len(unobserved) if unobserved else 0

print(f'{avg_fully:.2f} {avg_partially:.2f} {avg_unobserved:.2f}')
")
        
        read avg_fully avg_partially avg_unobserved <<< "$avg_metrics"
        
        # Output results
        echo ""
        echo "=========================================="
        echo "RESULTS FOR agent_region = $region"
        echo "=========================================="
        echo "Per-Agent Averages (averaged over $NUM_TRIALS trials):"
        echo "  Fully observed:     $avg_fully"
        echo "  Partially observed: $avg_partially"
        echo "  Unobserved:         $avg_unobserved"
        echo "=========================================="
        echo ""
        
        # Save to results file
        results_file="${RESULTS_DIR}/agent_region_${region}_results.txt"
        {
            echo "Agent Region: $region"
            echo "Number of Trials: $NUM_TRIALS"
            echo "Per-Agent Averages:"
            echo "  Fully observed:     $avg_fully"
            echo "  Partially observed: $avg_partially"
            echo "  Unobserved:         $avg_unobserved"
            echo ""
            echo "Individual Trial Results:"
            for i in "${!fully_values[@]}"; do
                echo "  Trial $((i+1)): Fully=${fully_values[$i]}, Partially=${partially_values[$i]}, Unobserved=${unobserved_values[$i]}"
            done
        } > "$results_file"
        
        # Append to summary file
        echo "$region $avg_fully $avg_partially $avg_unobserved" >> "${RESULTS_DIR}/summary.txt"
    else
        echo ""
        echo "ERROR: No valid metrics collected for agent_region = $region"
        echo ""
    fi
    
    # Clear arrays for next iteration
    unset fully_values
    unset partially_values
    unset unobserved_values
done

# Restore original config
cp "$BACKUP_CONFIG" "$CONFIG_FILE"
rm -f "$BACKUP_CONFIG"

# Print final summary
echo ""
echo "=========================================="
echo "FINAL SUMMARY - ALL AGENT REGIONS"
echo "=========================================="
echo "agent_region | Fully | Partially | Unobserved"
echo "-------------|-------|-----------|------------"

if [ -f "${RESULTS_DIR}/summary.txt" ]; then
    while read -r region fully partially unobserved; do
        printf "%12s | %5s | %9s | %10s\n" "$region" "$fully" "$partially" "$unobserved"
    done < "${RESULTS_DIR}/summary.txt"
fi

echo "=========================================="
echo ""
echo "All results saved to: $RESULTS_DIR"
echo "Summary file: ${RESULTS_DIR}/summary.txt"
echo "Original config restored: $CONFIG_FILE"
echo ""
echo "Experiment completed!"

