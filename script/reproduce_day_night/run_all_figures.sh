#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
SCRIPT_NAME=$(basename -- "$0")
MODULE_PREFIX=script.reproduce_day_night
ORIGINAL_ARGS=("$@")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_OUTPUT_ROOT=$REPO_ROOT/script/reproduce_day_night/output/figure_runs/$TIMESTAMP

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN=$PYTHON_BIN
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN=$REPO_ROOT/.venv/bin/python
else
    PYTHON_BIN=python3
fi

OUTPUT_ROOT=""
SKIP_WEIGHT_SWEEP="false"

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [options]

Relaunch the day-night figure suite with one shared parameter snapshot and
store the outputs in a dedicated campaign folder.

Options:
  --output-root DIR        Destination root for this campaign.
                           Default: $DEFAULT_OUTPUT_ROOT
  --python BIN             Python executable to use.
                           Default: .venv/bin/python when available, otherwise python3.
  --skip-weight-sweep      Skip the expensive Nash weight sweep.
  -h, --help               Show this help message.

Output layout inside the campaign folder:
  figures/basic_simulation/
  figures/game_theory/
  runs/game_theory/
  logs/
  simulation_parameters.json
  relaunch_command.sh
EOF
}

fail() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

require_value() {
    local option_name=$1
    if (($# < 2)); then
        fail "$option_name requires a value."
    fi
}

normalize_output_root() {
    mkdir -p -- "$1"
    cd -- "$1" && pwd
}

write_invocation_file() {
    local target_path=$1
    {
        printf '#!/usr/bin/env bash\n'
        printf 'set -euo pipefail\n\n'
        printf '%q' "$0"
        for argument in "${ORIGINAL_ARGS[@]}"; do
            printf ' %q' "$argument"
        done
        printf '\n'
    } > "$target_path"
}

check_python_modules() {
    "$PYTHON_BIN" - "$@" <<'PY'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Missing required Python modules: " + ", ".join(missing)
    )
PY
}

read_two_population_base_default() {
    local key=$1
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - "$key" <<'PY'
import sys

from script.reproduce_day_night.shared_config import TWO_POPULATION_SIMULATION_CONFIG

key = sys.argv[1]
value = TWO_POPULATION_SIMULATION_CONFIG["base"][key]
print(value)
PY
}

write_parameter_snapshot() {
    local output_path=$1
    local skip_weight_sweep=$2
    local payoff_mode=$3

    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - "$output_path" "$skip_weight_sweep" "$payoff_mode" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from script.reproduce_day_night.shared_config import (
    ONE_POPULATION_SIMULATION_CONFIG,
    TWO_POPULATION_SIMULATION_CONFIG,
)

output_path = Path(sys.argv[1])
skip_weight_sweep = sys.argv[2].lower() == "true"
payoff_mode = sys.argv[3]

payload = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "python_executable": sys.executable,
    "campaign": {
        "payoff_mode": payoff_mode,
        "skip_weight_sweep": skip_weight_sweep,
        "figure_scripts": [
            "script.reproduce_day_night.BasicSimulation.activity_const_heatmaps",
            "script.reproduce_day_night.BasicSimulation.activity_const_spread",
            "script.reproduce_day_night.BasicSimulation.sleep_pattern_heatmaps",
            "script.reproduce_day_night.BasicSimulation.sleep_pattern_spread",
            "script.reproduce_day_night.BasicSimulation.spread_diurnal_vs_nocturnal",
            "script.reproduce_day_night.BasicSimulation.spread_polyphasic_matutinal",
            "script.reproduce_day_night.BasicSimulation.spread_sleep_pattern_diffusion",
            "script.reproduce_day_night.GameTheory.payoff_initial_condition_comparison",
            "script.reproduce_day_night.GameTheory.payoff_weight_nash_heatmap",
            "script.reproduce_day_night.GameTheory.run_payoff_pipeline.sh",
        ],
    },
    "one_population_simulation_config": ONE_POPULATION_SIMULATION_CONFIG,
    "two_population_simulation_config": TWO_POPULATION_SIMULATION_CONFIG,
}

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

run_step() {
    local step_name=$1
    shift

    local log_path=$LOG_DIR/${step_name}.log

    printf '\n==> %s\n' "$step_name"
    printf '    '
    printf '%q ' "$@"
    printf '\n'

    "$@" 2>&1 | tee "$log_path"
}

copy_png_file() {
    local source_path=$1
    local destination_dir=$2

    if [[ ! -f "$source_path" ]]; then
        return
    fi

    mkdir -p -- "$destination_dir"
    cp -- "$source_path" "$destination_dir/"
}

copy_png_directory() {
    local source_dir=$1
    local destination_dir=$2
    local files=()

    if [[ ! -d "$source_dir" ]]; then
        return
    fi

    shopt -s nullglob
    files=("$source_dir"/*.png)
    shopt -u nullglob

    if ((${#files[@]} == 0)); then
        return
    fi

    mkdir -p -- "$destination_dir"
    cp -- "${files[@]}" "$destination_dir/"
}

while (($# > 0)); do
    case "$1" in
        --output-root)
            require_value "$1" "$@"
            OUTPUT_ROOT=$2
            shift 2
            ;;
        --python)
            require_value "$1" "$@"
            PYTHON_BIN=$2
            shift 2
            ;;
        --skip-weight-sweep)
            SKIP_WEIGHT_SWEEP="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown option: $1"
            ;;
    esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    fail "Python executable not found: $PYTHON_BIN"
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
    OUTPUT_ROOT=$DEFAULT_OUTPUT_ROOT
fi

OUTPUT_ROOT=$(normalize_output_root "$OUTPUT_ROOT")
LOG_DIR=$OUTPUT_ROOT/logs
FIGURES_DIR=$OUTPUT_ROOT/figures
BASIC_FIGURES_DIR=$FIGURES_DIR/basic_simulation
GAME_FIGURES_DIR=$FIGURES_DIR/game_theory
RUNS_DIR=$OUTPUT_ROOT/runs
GAME_RUNS_DIR=$RUNS_DIR/game_theory
PAYOFF_PIPELINE_DIR=$GAME_RUNS_DIR/payoff_pipeline
INITIAL_CONDITION_CACHE_DIR=$GAME_RUNS_DIR/initial_condition_payoff_matrices
WEIGHT_SWEEP_DIR=$GAME_RUNS_DIR/weight_nash_heatmap
SIMULATION_PARAMETERS_PATH=$OUTPUT_ROOT/simulation_parameters.json
PAYOFF_MODE=$(read_two_population_base_default payoff_mode)

mkdir -p -- \
    "$LOG_DIR" \
    "$BASIC_FIGURES_DIR" \
    "$GAME_FIGURES_DIR" \
    "$GAME_RUNS_DIR"

write_invocation_file "$OUTPUT_ROOT/relaunch_command.sh"
write_parameter_snapshot "$SIMULATION_PARAMETERS_PATH" "$SKIP_WEIGHT_SWEEP" "$PAYOFF_MODE"

printf 'Using Python executable: %s\n' "$PYTHON_BIN"
printf 'Writing campaign outputs under: %s\n' "$OUTPUT_ROOT"
printf 'Shared two-population payoff mode: %s\n' "$PAYOFF_MODE"

check_python_modules numpy matplotlib scipy nashpy

cd -- "$REPO_ROOT"

run_step activity_const_heatmaps \
    "$PYTHON_BIN" -m "$MODULE_PREFIX.BasicSimulation.activity_const_heatmaps" \
    --output-path "$BASIC_FIGURES_DIR/activity_const_heatmaps.png"

run_step activity_const_spread \
    "$PYTHON_BIN" -m "$MODULE_PREFIX.BasicSimulation.activity_const_spread" \
    --output-path "$BASIC_FIGURES_DIR/activity_const_spread.png"

run_step sleep_pattern_heatmaps \
    "$PYTHON_BIN" -m "$MODULE_PREFIX.BasicSimulation.sleep_pattern_heatmaps" \
    --output-path "$BASIC_FIGURES_DIR/sleep_pattern_heatmaps.png"

run_step sleep_pattern_spread \
    "$PYTHON_BIN" -m "$MODULE_PREFIX.BasicSimulation.sleep_pattern_spread" \
    --output-path "$BASIC_FIGURES_DIR/sleep_pattern_spread.png"

run_step spread_diurnal_vs_nocturnal \
    "$PYTHON_BIN" -m "$MODULE_PREFIX.BasicSimulation.spread_diurnal_vs_nocturnal" \
    --output-path "$BASIC_FIGURES_DIR/spread_diurnal_vs_nocturnal.png"

run_step spread_polyphasic_matutinal \
    "$PYTHON_BIN" -m "$MODULE_PREFIX.BasicSimulation.spread_polyphasic_matutinal" \
    --output-path "$BASIC_FIGURES_DIR/spread_polyphasic_matutinal.png"

run_step spread_sleep_pattern_diffusion \
    "$PYTHON_BIN" -m "$MODULE_PREFIX.BasicSimulation.spread_sleep_pattern_diffusion" \
    --output-path "$BASIC_FIGURES_DIR/spread_sleep_pattern_diffusion.png"

run_step payoff_pipeline \
    bash "$SCRIPT_DIR/GameTheory/run_payoff_pipeline.sh" \
    --output-dir "$PAYOFF_PIPELINE_DIR" \
    --python "$PYTHON_BIN" \
    --payoff-mode "$PAYOFF_MODE"

run_step payoff_initial_condition_comparison \
    "$PYTHON_BIN" -m "$MODULE_PREFIX.GameTheory.payoff_initial_condition_comparison" \
    --cache-dir "$INITIAL_CONDITION_CACHE_DIR" \
    --output-path "$GAME_FIGURES_DIR/payoff_initial_condition_comparison.png"

if [[ "$SKIP_WEIGHT_SWEEP" == "false" ]]; then
    run_step payoff_weight_nash_heatmap \
        "$PYTHON_BIN" -m "$MODULE_PREFIX.GameTheory.payoff_weight_nash_heatmap" \
        --output-dir "$WEIGHT_SWEEP_DIR" \
        --payoff-mode "$PAYOFF_MODE"
else
    printf '\nSkipping weight sweep as requested.\n'
fi

copy_png_file "$PAYOFF_PIPELINE_DIR/payoff_matrix.png" "$GAME_FIGURES_DIR/payoff_pipeline"
copy_png_file "$PAYOFF_PIPELINE_DIR/payoff_matrix_prey.png" "$GAME_FIGURES_DIR/payoff_pipeline"
copy_png_file "$PAYOFF_PIPELINE_DIR/payoff_matrix_predator.png" "$GAME_FIGURES_DIR/payoff_pipeline"
copy_png_directory "$PAYOFF_PIPELINE_DIR/mean_analysis" "$GAME_FIGURES_DIR/payoff_pipeline/mean_analysis"
copy_png_directory "$PAYOFF_PIPELINE_DIR/replicator_analysis" "$GAME_FIGURES_DIR/payoff_pipeline/replicator_analysis"

if [[ "$SKIP_WEIGHT_SWEEP" == "false" ]]; then
    copy_png_file "$WEIGHT_SWEEP_DIR/nash_consensus_components.png" "$GAME_FIGURES_DIR/weight_nash_heatmap"
    copy_png_file "$WEIGHT_SWEEP_DIR/nash_equilibrium_diagnostics.png" "$GAME_FIGURES_DIR/weight_nash_heatmap"
fi

printf '\nFigure campaign completed successfully.\n'
printf 'Main artifacts:\n'
printf '  %s\n' "$FIGURES_DIR"
printf '  %s\n' "$RUNS_DIR"
printf '  %s\n' "$SIMULATION_PARAMETERS_PATH"
printf '  %s\n' "$OUTPUT_ROOT/relaunch_command.sh"
printf '  %s\n' "$LOG_DIR"