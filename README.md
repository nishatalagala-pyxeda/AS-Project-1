# Low-power forecasting for HPC facilities

Explore whether compact models can forecast HPC accessory power and calculated waste heat accurately enough to support facility planning, while fitting on an ESP32-class microcontroller.

The potential applications are advance cooling and electrical planning, and coordination of heat recovery or storage. Operational savings have not been demonstrated. Forecast accuracy is the first step; hardware efficiency and the value of acting on forecasts remain research questions.

## Hypothesis

Compact tree-based models deployed on an ESP32 can forecast accessory power and calculated waste heat 30 minutes to six hours ahead with lower mean absolute error than persistence, while consuming less energy per forecast than an LSTM on the same hardware. Evaluate the complete feature-processing and inference pipeline, including memory, latency and joules per forecast, on previously unseen chronological data.

## Early indicators

A bounded sweep tested 120 forest configurations per target: Random Forest and Extra Trees, 1/4/16/64/128/256 trees, maximum depths 2/4/6/8/12, and target-only or all-telemetry features. Three expanding chronological development periods determined the selections. There were 720 development evaluations and 24 selected-model evaluations on the reused final period.

The following compact Extra Trees models were selected under a 32 KiB model-array budget, using a single model to predict all five horizons:

| Target | Trees / depth | Packed arrays | One-hour MAE | One-hour reduction vs persistence | Six-hour reduction vs persistence |
| --- | --- | --- | --- | --- | --- |
| Accessory power | 16 / 6 | 31,776 bytes (31.0 KiB) | 42.3 kW | 10.1% | 16.0% |
| Calculated waste heat | 16 / 4 | 8,988 bytes (8.8 KiB) | 1.479 MW | 5.6% | 17.6% |

Larger shared forests added little accuracy. Earlier comparisons found no advantage from the tested LSTM, GRU and temporal convolutional configurations over selected simpler approaches; boosted trees and Ridge regression often remained more accurate. Persistence is a sensible ten-minute baseline.

The selected forests were exported to C, compiled on the host, and checked against Python predictions. **No ESP32 energy, latency, or total memory measurements have been made.** Array sizes exclude firmware, feature buffers, runtime and communications. The C exports accept precomputed features; a complete embedded feature-processing pipeline is not implemented.

## Dataset and evaluation limits

The input is `Frontier HPC & Facility Data.xlsx`: 49,869 observations from 2023, nominally every ten minutes, with 17 numeric fields plus timestamps. Measurements include compute/accessory/total electrical power, cooling supply and return temperatures, loop flow, PUE, and calculated waste heat. Approximately 5.1% of expected timestamps are missing. Workload schedules, weather and maintenance records are absent.

The original workbook is not included or modified. Place your copy outside the repository or in an ignored location. The analyzed file has SHA-256:

```text
56a48dd66382896890dfc2e4940ec90f894bb85795ee8e85ea2708dda05796bf
```

October–December was inspected in earlier experiments. Its reuse here is exploratory confirmation, **not a fresh independent holdout**. New chronological data is needed to establish generalization. Peak-demand forecasting was weaker than average performance. Waste heat is calculated from temperature and flow using the workbook's thermal assumptions; it is not an independent heat-meter measurement. Forecasting benefits do not establish operational energy savings.

The broad benchmark and forest sweep use common complete origins, past-only features, train-only scaling, and horizons of 10/30/60/120/360 minutes. Missing windows are excluded rather than imputed. Forest selection uses mean relative MAE across June–July, August and September scoring periods, with equal period weights. The final period is not used to retune the selections. See manifests and experiment source for full definitions.

## Repository contents

| Directory / file | Purpose |
| --- | --- |
| `assess_dataset.py` | Read-only workbook assessment and physical/algebraic consistency checks |
| `frontier_experiment/` | Initial total-power LSTM, boosted-tree and persistence comparison |
| `frontier_targets_experiment/` | Screening of six additional prediction tasks |
| `frontier_full_benchmark/` | Broader model comparisons, rolling evaluation and uncertainty experiments |
| `frontier_forest_sweep/` | Forest capacity/storage sweep, verification, reports and selected C exports |

Start with [the forest report](frontier_forest_sweep/report.html) or [the broader benchmark report](frontier_full_benchmark/report.html). HTML reports and PNG plots can be viewed locally. Metrics, selection records, manifests and summary CSV files are committed. Large fitted models, prediction archives, host binaries and logs are ignored and must be regenerated for checks that require them.

The compact C exports are:

- [Accessory power](frontier_forest_sweep/accessory_power/confirmation/ET_all_d6_n16/forest.c), with [feature order and scaling](frontier_forest_sweep/accessory_power/confirmation/ET_all_d6_n16/metadata.json).
- [Waste heat](frontier_forest_sweep/waste_heat/confirmation/ET_own_d4_n16/forest.c), with [feature order and scaling](frontier_forest_sweep/waste_heat/confirmation/ET_own_d4_n16/metadata.json).

These files each define `forest_predict`; compile them separately or rename the exported functions when integrating both in one firmware build. Static arrays and inference are provided, not an ESP32 firmware application.

## Reproduce

Use Python 3.12 and install the recorded versions:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export FRONTIER_SOURCE="/path/to/Frontier HPC & Facility Data.xlsx"
```

Run the broader benchmark in target order, writing fresh results under the ignored `.runs` directory so committed metric files do not trigger resume skips:

```bash
python frontier_full_benchmark/run_benchmark.py --target accessory_power --source "$FRONTIER_SOURCE" --output .runs/frontier_full_benchmark
python frontier_full_benchmark/run_benchmark.py --target waste_heat --source "$FRONTIER_SOURCE" --output .runs/frontier_full_benchmark
```

Then run the forest sweep and build its report:

```bash
export FRONTIER_FOREST_OUTPUT="$PWD/.runs/frontier_forest_sweep"
python frontier_forest_sweep/run_sweep.py
python frontier_forest_sweep/build_report.py
```

The sweep requires a host C compiler (`cc`) to validate its exports. It reuses the shared data loader in `frontier_full_benchmark/data.py`. The report builder needs regenerated prediction archives and the sibling broad-benchmark results for verification; it does not work from the committed metric summaries alone. These commands preserve the committed reports and metrics. Existing completed configurations in `.runs` are skipped; move the relevant generated target folders out of `.runs` before a clean rerun. Historical `reproduce.txt` files and manifests retain the original machine paths as provenance.

For the initial experiments:

```bash
python assess_dataset.py
python frontier_experiment/run_experiment.py --source "$FRONTIER_SOURCE"
python frontier_targets_experiment/run_targets.py --task all --source "$FRONTIER_SOURCE"
```

The scripts retain the original local dependency-directory fallback; a normal virtual environment with the pinned packages also works. Saved fitted Python model files are regenerated locally and should only be loaded from trusted runs.

## Next experiment

Implement rolling features and inference on an ESP32, compare compact forests with Ridge and persistence (and a small LSTM for the energy hypothesis), and measure complete-pipeline memory, latency and joules per forecast. Validate on new chronological data and peak conditions before testing whether forecast-driven operation improves cooling or heat utilization.
