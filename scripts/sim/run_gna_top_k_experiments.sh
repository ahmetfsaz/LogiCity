#!/bin/bash
# Experiment script to test different gna_top_k values with multiple random seeds
# Tests gna_top_k from 1 to 10 with 20 trials each

# source /opt/conda/etc/profile.d/conda.sh
source /opt/anaconda3/etc/profile.d/conda.sh
conda init
conda activate logicity

# Configuration
CONFIG_FILE="config/tasks/sim/expert.yaml"
BACKUP_CONFIG="${CONFIG_FILE}.backup"
MAX_STEPS=100
NUM_TRIALS=5
LOG_DIR="log_sim"

# GNA top-k values to test (from 0 to 10)
GNA_TOP_K_VALUES=(0 1 2 3 4 5 6 7 8 9 10)

# Create results directory
RESULTS_DIR="gna_top_k_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Backup original config
cp "$CONFIG_FILE" "$BACKUP_CONFIG"

echo "=========================================="
echo "GNA Top-K Experiment - Subrule Analysis"
echo "=========================================="
echo "Number of trials per gna_top_k: $NUM_TRIALS"
echo "GNA top-k values to test: ${GNA_TOP_K_VALUES[@]}"
echo "Max steps per trial: $MAX_STEPS"
echo "Results directory: $RESULTS_DIR"
echo "=========================================="
echo ""

# Function to update gna_top_k in config file
update_gna_top_k() {
    local top_k=$1
    # Use sed to replace the gna_top_k line
    sed -i.tmp "s/gna_top_k:.*/gna_top_k: $top_k/" "$CONFIG_FILE"
    rm -f "${CONFIG_FILE}.tmp"
}

# Function to extract metrics from log file
extract_metrics() {
    local log_file=$1
    # Extract the last occurrence of "Per-Agent Averages" line
    grep "Per-Agent Averages" "$log_file" | tail -1 | \
        awk -F'Fully: |, Partially: |, Unobserved: | \\(Agents' '{print $2, $3, $4}'
}

# Function to extract informativeness from log file
extract_informativeness() {
    local log_file=$1
    # Extract "Average Normalized Informativeness" line and get the numeric value
    grep "Average Normalized Informativeness:" "$log_file" | tail -1 | \
        awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+$/) {print $i; exit}}'
}

# Main experiment loop
for top_k in "${GNA_TOP_K_VALUES[@]}"; do
    echo "=========================================="
    echo "Testing gna_top_k = $top_k"
    echo "=========================================="
    
    # Update config file
    update_gna_top_k "$top_k"
    echo "Updated $CONFIG_FILE with gna_top_k: $top_k"
    
    # Arrays to store metrics across trials
    declare -a fully_values
    declare -a partially_values
    declare -a unobserved_values
    declare -a informativeness_values
    
    # Run trials
    for seed in $(seq 0 $((NUM_TRIALS - 1))); do
        EXPNAME="expert_topk${top_k}_seed${seed}"
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
        informativeness=$(extract_informativeness "$LOG_FILE")
        
        if [ -n "$metrics" ]; then
            read fully partially unobserved <<< "$metrics"
            fully_values+=("$fully")
            partially_values+=("$partially")
            unobserved_values+=("$unobserved")
            
            # Add informativeness if available
            if [ -n "$informativeness" ]; then
                informativeness_values+=("$informativeness")
                echo "Done (Fully: $fully, Partially: $partially, Unobserved: $unobserved, Info: $informativeness)"
            else
                informativeness_values+=("0.0")
                echo "Done (Fully: $fully, Partially: $partially, Unobserved: $unobserved, Info: N/A)"
            fi
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
        informativeness_str=$(IFS=,; echo "${informativeness_values[*]}")
        
        avg_metrics=$(python3 -c "
import sys
fully = [${fully_str}]
partially = [${partially_str}]
unobserved = [${unobserved_str}]
informativeness = [${informativeness_str}]

avg_fully = sum(fully) / len(fully) if fully else 0
avg_partially = sum(partially) / len(partially) if partially else 0
avg_unobserved = sum(unobserved) / len(unobserved) if unobserved else 0
avg_informativeness = sum(informativeness) / len(informativeness) if informativeness else 0

print(f'{avg_fully:.2f} {avg_partially:.2f} {avg_unobserved:.2f} {avg_informativeness:.3f}')
")
        
        read avg_fully avg_partially avg_unobserved avg_informativeness <<< "$avg_metrics"
        
        # Output results
        echo ""
        echo "=========================================="
        echo "RESULTS FOR gna_top_k = $top_k"
        echo "=========================================="
        echo "Per-Agent Averages (averaged over $NUM_TRIALS trials):"
        echo "  Fully observed:     $avg_fully"
        echo "  Partially observed: $avg_partially"
        echo "  Unobserved:         $avg_unobserved"
        echo "  Informativeness:    $avg_informativeness ($(python3 -c "print(f'{${avg_informativeness}*100:.1f}%')"))"
        echo "=========================================="
        echo ""
        
        # Save to results file
        results_file="${RESULTS_DIR}/gna_top_k_${top_k}_results.txt"
        {
            echo "GNA Top-K: $top_k"
            echo "Number of Trials: $NUM_TRIALS"
            echo "Per-Agent Averages:"
            echo "  Fully observed:     $avg_fully"
            echo "  Partially observed: $avg_partially"
            echo "  Unobserved:         $avg_unobserved"
            echo "  Informativeness:    $avg_informativeness"
            echo ""
            echo "Individual Trial Results:"
            for i in "${!fully_values[@]}"; do
                info_val="${informativeness_values[$i]:-0.0}"
                echo "  Trial $((i+1)): Fully=${fully_values[$i]}, Partially=${partially_values[$i]}, Unobserved=${unobserved_values[$i]}, Info=${info_val}"
            done
        } > "$results_file"
        
        # Append to summary file
        echo "$top_k $avg_fully $avg_partially $avg_unobserved $avg_informativeness" >> "${RESULTS_DIR}/summary.txt"
    else
        echo ""
        echo "ERROR: No valid metrics collected for gna_top_k = $top_k"
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
echo "FINAL SUMMARY - ALL GNA TOP-K VALUES"
echo "=========================================="
echo "gna_top_k | Fully | Partially | Unobserved | Informativeness"
echo "----------|-------|-----------|------------|----------------"

if [ -f "${RESULTS_DIR}/summary.txt" ]; then
    while read -r top_k fully partially unobserved informativeness; do
        info_pct=$(python3 -c "print(f'{${informativeness}*100:.1f}%')")
        printf "%9s | %5s | %9s | %10s | %6s (%s)\n" "$top_k" "$fully" "$partially" "$unobserved" "$informativeness" "$info_pct"
    done < "${RESULTS_DIR}/summary.txt"
fi

echo "=========================================="
echo ""
echo "All results saved to: $RESULTS_DIR"
echo "Summary file: ${RESULTS_DIR}/summary.txt"
echo "Original config restored: $CONFIG_FILE"
echo ""
echo "Experiment completed!"

