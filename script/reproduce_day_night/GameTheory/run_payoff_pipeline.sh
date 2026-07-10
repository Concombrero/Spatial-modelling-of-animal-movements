#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
SCRIPT_NAME=$(basename -- "$0")
MODULE_PREFIX=script.reproduce_day_night.GameTheory

ORIGINAL_ARGS=("$@")

PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_DIR=""
MEAN_X_AXES="w1,w2,cycle1,cycle2"
MEAN_SHOW_VARIANCE="false"
REPLICATOR_TIME_SPAN="40"
REPLICATOR_TIME_STEPS="800"
REPLICATOR_PLOT_STYLE="line"
PAYOFF_MODE="overlap"
HEATMAP_PREY_SET="false"
HEATMAP_PREDATOR_SET="false"
PAYOFF_ARGS=()

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME --output-dir DIR [options]

Run the full payoff-analysis pipeline in one chosen output folder:
  1. payoff_matrix.py
  2. payoff_mean_analysis.py
  3. payoff_minmax_maxmin.py
  4. payoff_nash_equilibrium.py
  5. payoff_replicator_analysis.py

General options:
  --output-dir DIR                Destination folder for all outputs. Required.
  --python BIN                    Python executable to use. Default: python.
  --mean-x-axes LIST              Comma-separated axes for mean analysis.
                                  Default: w1,w2,cycle1,cycle2.
  --mean-show-variance BOOL       true/false for variance subplot. Default: false.
  --replicator-time-span FLOAT    Replicator final time. Default: 40.
  --replicator-time-steps INT     Replicator sample count. Default: 800.
  --replicator-plot-style STYLE   line or stacked. Default: line.
    --payoff-mode MODE              overlap or population-integral. Default: overlap.

Payoff-matrix options:
    --strategy-codes LIST          Comma-separated subset of activity codes.
  --t-sunset FLOAT
  --weights W1 W2
  --sight-radius FLOAT
  --prey-sight-radius FLOAT
  --predator-sight-radius FLOAT
  --smell-radius FLOAT
  --prey-smell-radius FLOAT
  --predator-smell-radius FLOAT
  --number-of-points INT
  --dt FLOAT
  --number-of-cycles INT
  --observation-window FLOAT
  --prey-growth FLOAT
  --predator-decay FLOAT
  --predation-rate FLOAT
  --conversion-rate FLOAT
  --chi11 FLOAT
  --chi12 FLOAT
  --chi21 FLOAT
  --chi22 FLOAT
  --diffusion D1 D2
  --initial-centers X1 X2
  --initial-width FLOAT
  --heatmap-prey CODE
  --heatmap-predator CODE
  --max-workers INT

See $SCRIPT_DIR/run_payoff_pipeline.README.md for details and examples.
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

require_two_values() {
    local option_name=$1
    if (($# < 3)); then
        fail "$option_name requires two values."
    fi
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

normalize_output_dir() {
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

while (($# > 0)); do
    case "$1" in
        --output-dir)
            require_value "$1" "$@"
            OUTPUT_DIR=$2
            shift 2
            ;;
        --python)
            require_value "$1" "$@"
            PYTHON_BIN=$2
            shift 2
            ;;
        --mean-x-axes)
            require_value "$1" "$@"
            MEAN_X_AXES=$2
            shift 2
            ;;
        --mean-show-variance)
            require_value "$1" "$@"
            MEAN_SHOW_VARIANCE=$2
            shift 2
            ;;
        --replicator-time-span)
            require_value "$1" "$@"
            REPLICATOR_TIME_SPAN=$2
            shift 2
            ;;
        --replicator-time-steps)
            require_value "$1" "$@"
            REPLICATOR_TIME_STEPS=$2
            shift 2
            ;;
        --replicator-plot-style)
            require_value "$1" "$@"
            REPLICATOR_PLOT_STYLE=$2
            shift 2
            ;;
        --payoff-mode)
            require_value "$1" "$@"
            PAYOFF_MODE=$2
            shift 2
            ;;
        --weights|--diffusion|--initial-centers)
            require_two_values "$1" "$@"
            PAYOFF_ARGS+=("$1" "$2" "$3")
            shift 3
            ;;
        --strategy-codes|--t-sunset|--sight-radius|--prey-sight-radius|--predator-sight-radius|--smell-radius|--prey-smell-radius|--predator-smell-radius|--number-of-points|--dt|--number-of-cycles|--observation-window|--prey-growth|--predator-decay|--predation-rate|--conversion-rate|--chi11|--chi12|--chi21|--chi22|--initial-width|--max-workers)
            require_value "$1" "$@"
            PAYOFF_ARGS+=("$1" "$2")
            shift 2
            ;;
        --heatmap-prey)
            require_value "$1" "$@"
            HEATMAP_PREY_SET="true"
            PAYOFF_ARGS+=("$1" "$2")
            shift 2
            ;;
        --heatmap-predator)
            require_value "$1" "$@"
            HEATMAP_PREDATOR_SET="true"
            PAYOFF_ARGS+=("$1" "$2")
            shift 2
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

if [[ -z "$OUTPUT_DIR" ]]; then
    fail "--output-dir is required."
fi

if [[ "$HEATMAP_PREY_SET" != "$HEATMAP_PREDATOR_SET" ]]; then
    fail "--heatmap-prey and --heatmap-predator must be provided together."
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    fail "Python executable not found: $PYTHON_BIN"
fi

IFS=',' read -r -a RAW_MEAN_AXES <<< "$MEAN_X_AXES"
MEAN_AXES=()
for axis in "${RAW_MEAN_AXES[@]}"; do
    axis=${axis//[[:space:]]/}
    if [[ -n "$axis" ]]; then
        MEAN_AXES+=("$axis")
    fi
done

if ((${#MEAN_AXES[@]} == 0)); then
    fail "--mean-x-axes must contain at least one axis."
fi

OUTPUT_DIR=$(normalize_output_dir "$OUTPUT_DIR")
LOG_DIR=$OUTPUT_DIR/logs
MEAN_OUTPUT_DIR=$OUTPUT_DIR/mean_analysis
REPLICATOR_OUTPUT_DIR=$OUTPUT_DIR/replicator_analysis
PAYOFF_MATRIX_PATH=$OUTPUT_DIR/payoff_matrix.csv
PREY_PAYOFF_MATRIX_PATH=$OUTPUT_DIR/payoff_matrix_prey.csv
PREDATOR_PAYOFF_MATRIX_PATH=$OUTPUT_DIR/payoff_matrix_predator.csv

mkdir -p -- "$LOG_DIR" "$MEAN_OUTPUT_DIR" "$REPLICATOR_OUTPUT_DIR"

write_invocation_file "$OUTPUT_DIR/pipeline_command.sh"

printf 'Using Python executable: %s\n' "$PYTHON_BIN"
printf 'Writing pipeline outputs under: %s\n' "$OUTPUT_DIR"

check_python_modules numpy matplotlib scipy nashpy

cd -- "$REPO_ROOT"

run_step payoff_matrix \
    "$PYTHON_BIN" -m "$MODULE_PREFIX.payoff_matrix" \
    --output-dir "$OUTPUT_DIR" \
    --payoff-mode "$PAYOFF_MODE" \
    "${PAYOFF_ARGS[@]}"

for axis in "${MEAN_AXES[@]}"; do
    axis_slug=${axis//[^[:alnum:]_-]/_}
    if [[ "$PAYOFF_MODE" == "overlap" ]]; then
        run_step "mean_${axis_slug}" \
            "$PYTHON_BIN" -m "$MODULE_PREFIX.payoff_mean_analysis" \
            --x-axis "$axis" \
            --payoff-dir "$OUTPUT_DIR" \
            --output "$MEAN_OUTPUT_DIR/mean_vs_${axis_slug}.png" \
            --show-variance "$MEAN_SHOW_VARIANCE"
    else
        for payoff_player in prey predator; do
            run_step "mean_${axis_slug}_${payoff_player}" \
                "$PYTHON_BIN" -m "$MODULE_PREFIX.payoff_mean_analysis" \
                --x-axis "$axis" \
                --payoff-dir "$OUTPUT_DIR" \
                --output "$MEAN_OUTPUT_DIR/mean_vs_${axis_slug}_${payoff_player}.png" \
                --show-variance "$MEAN_SHOW_VARIANCE" \
                --payoff-player "$payoff_player"
        done
    fi
done

run_step minmax \
    "$PYTHON_BIN" -m "$MODULE_PREFIX.payoff_minmax_maxmin" \
    --payoff-matrix "$OUTPUT_DIR"

run_step nash \
    "$PYTHON_BIN" -m "$MODULE_PREFIX.payoff_nash_equilibrium" \
    --payoff-matrix "$OUTPUT_DIR"

run_step replicator \
    "$PYTHON_BIN" -m "$MODULE_PREFIX.payoff_replicator_analysis" \
    --payoff-matrix "$OUTPUT_DIR" \
    --time-span "$REPLICATOR_TIME_SPAN" \
    --time-steps "$REPLICATOR_TIME_STEPS" \
    --output-dir "$REPLICATOR_OUTPUT_DIR" \
    --plot-style "$REPLICATOR_PLOT_STYLE"

printf '\nPipeline completed successfully.\n'
printf 'Main artifacts:\n'
if [[ "$PAYOFF_MODE" == "overlap" ]]; then
    printf '  %s\n' "$PAYOFF_MATRIX_PATH"
else
    printf '  %s\n' "$PREY_PAYOFF_MATRIX_PATH"
    printf '  %s\n' "$PREDATOR_PAYOFF_MATRIX_PATH"
fi
printf '  %s\n' "$OUTPUT_DIR/payoff_minmax_maxmin.json"
printf '  %s\n' "$OUTPUT_DIR/payoff_nash_equilibrium.json"
printf '  %s\n' "$MEAN_OUTPUT_DIR"
printf '  %s\n' "$REPLICATOR_OUTPUT_DIR"
printf '  %s\n' "$LOG_DIR"