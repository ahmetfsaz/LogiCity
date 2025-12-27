# Agent Region Experiment Scripts

This directory contains scripts for running systematic experiments to analyze the effect of `agent_region` parameter on sub-rule observability.

## Files

- `run_agent_region_experiments.sh` - Main experiment script that runs multiple trials across different agent_region values
- `analyze_experiment_results.py` - Python script to analyze and visualize experimental results
- `run_sim_expert.sh` - Single run script for expert mode (original)

## Quick Start

### Running Experiments

To run the full experiment suite (tests agent_region from 70 to 200 with 20 trials each):

```bash
cd /path/to/LogiCity
./scripts/sim/run_agent_region_experiments.sh
```

This will:
- Test agent_region values: 70, 80, 90, ..., 200 (14 different values)
- Run 20 trials per agent_region value with different random seeds
- Average the "Per-Agent Averages" metrics across all 20 trials
- Save results to a timestamped directory `experiment_results_YYYYMMDD_HHMMSS/`
- Print a summary table at the end

**Expected runtime**: ~280 trials × execution time per trial

### Analyzing Results

After the experiment completes, analyze the results:

```bash
# Basic analysis (prints table and statistics)
python3 scripts/sim/analyze_experiment_results.py experiment_results_YYYYMMDD_HHMMSS/

# Generate plots (requires matplotlib)
python3 scripts/sim/analyze_experiment_results.py experiment_results_YYYYMMDD_HHMMSS/ --plot

# Export to CSV
python3 scripts/sim/analyze_experiment_results.py experiment_results_YYYYMMDD_HHMMSS/ --csv

# Do everything
python3 scripts/sim/analyze_experiment_results.py experiment_results_YYYYMMDD_HHMMSS/ --plot --csv
```

## Output Files

### Experiment Results Directory Structure

```
experiment_results_YYYYMMDD_HHMMSS/
├── summary.txt                          # Summary of all agent_region values
├── agent_region_70_results.txt          # Detailed results for agent_region=70
├── agent_region_80_results.txt          # Detailed results for agent_region=80
├── ...
├── agent_region_200_results.txt         # Detailed results for agent_region=200
├── results.csv                          # CSV export (if --csv used)
└── experiment_plots.png                 # Visualization (if --plot used)
```

### Summary File Format

The `summary.txt` file contains one line per agent_region:
```
70 2.45 7.89 2.66
80 2.67 7.45 2.88
...
```
Format: `agent_region fully_observed partially_observed unobserved`

### Individual Results Files

Each `agent_region_XXX_results.txt` contains:
- Agent region value
- Number of trials
- Averaged metrics (Fully, Partially, Unobserved)
- Individual trial results for reproducibility

## Customizing Experiments

### Modifying Agent Region Range

Edit `run_agent_region_experiments.sh` and change the `AGENT_REGIONS` array:

```bash
# Test only specific values
AGENT_REGIONS=(70 100 150 200)

# Test with finer granularity
AGENT_REGIONS=($(seq 70 5 200))  # 70, 75, 80, ..., 200
```

### Changing Number of Trials

Edit the `NUM_TRIALS` variable:

```bash
NUM_TRIALS=10  # Run 10 trials instead of 20
```

### Modifying Max Steps

Edit the `MAX_STEPS` variable:

```bash
MAX_STEPS=200  # Run longer simulations
```

## Understanding the Metrics

### Per-Agent Averages

These metrics are extracted from the subrule analysis and represent the average number of sub-rules per agent that fall into each category:

- **Fully Observed**: All required entity types for the sub-rule are present in the agent's FOV or GNA broadcast
- **Partially Observed**: Some (but not all) required entity types are present
- **Unobserved**: None of the required entity types are present

### Agent Region Parameter

The `agent_region` parameter controls the spatial region size for agent interactions. Higher values typically mean:
- Larger field of view or interaction area
- More entities potentially visible
- Higher observability of sub-rules

## Troubleshooting

### Script Not Finding main.py

Ensure you're running from the LogiCity root directory:
```bash
cd /Users/ahmetfaruksaz/GitHub/CityLogi/LogiCity
./scripts/sim/run_agent_region_experiments.sh
```

### Conda Environment Issues

The script assumes conda is installed at `/opt/anaconda3/`. If your installation is elsewhere, edit the script:

```bash
# Change this line in run_agent_region_experiments.sh
source /YOUR/CONDA/PATH/etc/profile.d/conda.sh
```

### Missing Dependencies for Plotting

Install matplotlib:
```bash
conda activate logicity
pip install matplotlib
```

### Experiment Interrupted

If the experiment is interrupted, you can:
1. Check which agent_region values completed in the results directory
2. Manually edit `AGENT_REGIONS` array to skip completed values
3. Re-run the script

The original config file is automatically restored from backup even if interrupted.

## Example Output

```
==========================================
RESULTS FOR agent_region = 120
==========================================
Per-Agent Averages (averaged over 20 trials):
  Fully observed:     3.05
  Partially observed: 6.64
  Unobserved:         3.31
==========================================
```

Final summary:
```
==========================================
FINAL SUMMARY - ALL AGENT REGIONS
==========================================
agent_region | Fully | Partially | Unobserved
-------------|-------|-----------|------------
          70 |  2.15 |      7.92 |       2.93
          80 |  2.34 |      7.68 |       2.98
         ...
         200 |  3.89 |      5.23 |       3.88
==========================================
```

## Notes

- Each trial uses a different random seed (0 to NUM_TRIALS-1) for reproducibility
- The original `config/tasks/sim/expert.yaml` is backed up and restored after experiments
- All trial logs are saved to `log_sim/` directory
- Experiments can take several hours depending on NUM_TRIALS and MAX_STEPS

