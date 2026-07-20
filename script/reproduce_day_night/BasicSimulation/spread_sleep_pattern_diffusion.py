import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
import sys
import traceback

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.reproduce_day_night.shared_config import (
    ONE_POPULATION_SIMULATION_CONFIG,
    activity_regimes_for_codes,
    apply_plot_typography,
    build_periodic_lighting_regime,
    resolve_experiment_config,
)
from script.reproduce_day_night.Solver import (
    DayNightModel1D,
    compute_spread_indicator,
    gaussian_initial_condition,
)


apply_plot_typography()


EXPERIMENT_CONFIG = resolve_experiment_config(
    ONE_POPULATION_SIMULATION_CONFIG,
    "spread_sleep_pattern_diffusion",
)
NUMBER_OF_POINTS = EXPERIMENT_CONFIG["number_of_points"]
NUMBER_OF_POPULATIONS = EXPERIMENT_CONFIG["number_of_populations"]
NUMBER_OF_CYCLES = EXPERIMENT_CONFIG["number_of_cycles"]
CYCLE_PERIOD = EXPERIMENT_CONFIG["cycle_period"]
TOTAL_TIME = NUMBER_OF_CYCLES * CYCLE_PERIOD
OBSERVATION_WINDOW = EXPERIMENT_CONFIG["observation_window"]
DT = EXPERIMENT_CONFIG["dt"]
COEFFICIENT_ATTRACTION = np.array(
    EXPERIMENT_CONFIG["coefficient_attraction"],
    dtype=float,
)
BASE_COEFFICIENT_DIFFUSION = np.array(
    EXPERIMENT_CONFIG["coefficient_diffusion"],
    dtype=float,
)
SIGHT_RADIUS = EXPERIMENT_CONFIG["sight_radius"]
SMELL_RADIUS = EXPERIMENT_CONFIG["smell_radius"]
SIGHT_WEIGHTS = EXPERIMENT_CONFIG["weights"]
MAX_WORKERS = EXPERIMENT_CONFIG["max_workers"]
DAY_START = EXPERIMENT_CONFIG["day_start"]
T_SUNSET = EXPERIMENT_CONFIG["t_sunset"]
DIFFUSION_SCALES = EXPERIMENT_CONFIG["diffusion_scales"]
CASE_TIMEOUT_SECONDS = EXPERIMENT_CONFIG["case_timeout_seconds"]
RETRY_POINT_SCALES = EXPERIMENT_CONFIG["retry_point_scales"]
RETRY_DT_SCALES = EXPERIMENT_CONFIG["retry_dt_scales"]
MIN_RETRY_NUMBER_OF_POINTS = EXPERIMENT_CONFIG["min_retry_number_of_points"]
ACTIVITY_REGIMES = activity_regimes_for_codes(EXPERIMENT_CONFIG["activity_codes"])
OUTPUT_PATH = (
    Path(__file__).resolve().parents[3]
    / "article"
    / "figures"
    / "sleep_pattern_diffusion_spread.png"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the normalized spread indicator Psi for all circadian "
            "activity patterns at fixed t_sunset=0.5 and compare three "
            "diffusion levels D/10, D, and 10D in a single article figure."
        )
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=list(SIGHT_WEIGHTS),
        help="Sight weights w to evaluate.",
    )
    parser.add_argument(
        "--diffusion-scales",
        nargs="+",
        type=float,
        default=list(DIFFUSION_SCALES),
        help="Multipliers applied to the baseline diffusion coefficient.",
    )
    parser.add_argument(
        "--t-sunset",
        type=float,
        default=T_SUNSET,
        help="Daylight proportion t_sunset in the interval [0, 1].",
    )
    parser.add_argument(
        "--number-of-points",
        type=int,
        default=NUMBER_OF_POINTS,
        help="Number of spatial grid points.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DT,
        help="Output time step used by the solver.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Maximum number of concurrent simulation tasks across all cases.",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=CASE_TIMEOUT_SECONDS,
        help=(
            "Maximum wall-clock time allowed for one simulation attempt before "
            "it is terminated and retried with a lighter setup."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Path of the saved figure.",
    )
    return parser.parse_args()


def build_lighting_regime(t_sunset, dt):
    return build_periodic_lighting_regime(
        t_sunset,
        dt=dt,
        cycle_period=CYCLE_PERIOD,
        day_start=DAY_START,
    )


def build_solver(
    sight_weight,
    lighting_regime,
    activity_regime,
    diffusion_scale,
    number_of_points,
    dt,
):
    return DayNightModel1D(
        a_border=0.0,
        b_border=1.0,
        number_of_points=number_of_points,
        total_time=TOTAL_TIME,
        dt=dt,
        initial_condition=gaussian_initial_condition,
        coefficient_attraction=COEFFICIENT_ATTRACTION,
        coefficient_diffusion=BASE_COEFFICIENT_DIFFUSION * float(diffusion_scale),
        cycle_period=CYCLE_PERIOD,
        number_of_population=NUMBER_OF_POPULATIONS,
        day_start=lighting_regime["day_start"],
        day_end=lighting_regime["day_end"],
        time_input_mode="phase",
        activity_mode="always",
        activity_periods=activity_regime["periods"],
        sight_weight=sight_weight,
        sight_radius=SIGHT_RADIUS,
        smell_radius=SMELL_RADIUS,
    )


def compute_psi(model, observation_window=OBSERVATION_WINDOW, population_index=0):
    return compute_spread_indicator(
        model,
        observation_window,
        population_index=population_index,
    )


def build_attempt_parameters(number_of_points, dt):
    attempts = []

    for point_scale, dt_scale in zip(RETRY_POINT_SCALES, RETRY_DT_SCALES):
        attempt_points = int(round(number_of_points * float(point_scale)))
        if point_scale < 1.0:
            attempt_points = max(MIN_RETRY_NUMBER_OF_POINTS, attempt_points)
        attempt_points = max(2, min(int(number_of_points), attempt_points))
        attempt_dt = float(dt) * float(dt_scale)
        attempt = (attempt_points, attempt_dt)
        if attempt not in attempts:
            attempts.append(attempt)

    return attempts


def run_single_case(
    diffusion_index,
    regime_index,
    weight_index,
    lighting_regime,
    activity_regime,
    sight_weight,
    diffusion_scale,
    number_of_points,
    dt,
):
    model = build_solver(
        sight_weight,
        lighting_regime,
        activity_regime,
        diffusion_scale,
        number_of_points,
        dt,
    )
    model.solve()
    psi_value = compute_psi(model)
    return diffusion_index, regime_index, weight_index, psi_value


def run_single_case_in_subprocess(result_queue, case_arguments):
    try:
        result_queue.put({"status": "ok", "result": run_single_case(*case_arguments)})
    except Exception:
        result_queue.put({"status": "error", "message": traceback.format_exc()})


def summarize_error_message(message):
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    if not lines:
        return str(message)
    return lines[-1]


def run_case_with_timeout(case_arguments, timeout_seconds):
    context = get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=run_single_case_in_subprocess,
        args=(result_queue, case_arguments),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(2.0)
        if process.is_alive():
            process.kill()
            process.join()
        result_queue.close()
        result_queue.join_thread()
        return {"status": "timeout", "message": f"Timed out after {timeout_seconds:g}s."}

    try:
        payload = result_queue.get_nowait()
    except Empty:
        payload = {
            "status": "error",
            "message": (
                "Simulation worker exited without returning a result "
                f"(exit code {process.exitcode})."
            ),
        }
    finally:
        result_queue.close()
        result_queue.join_thread()

    return payload


def run_case_with_retries(
    case_spec,
    lighting_regime,
    number_of_points,
    dt,
    case_timeout_seconds,
):
    (
        diffusion_index,
        regime_index,
        weight_index,
        diffusion_scale,
        activity_regime,
        sight_weight,
    ) = case_spec
    attempt_parameters = build_attempt_parameters(number_of_points, dt)
    failure_messages = []

    for attempt_index, (attempt_points, attempt_dt) in enumerate(
        attempt_parameters,
        start=1,
    ):
        payload = run_case_with_timeout(
            (
                diffusion_index,
                regime_index,
                weight_index,
                lighting_regime,
                activity_regime,
                sight_weight,
                diffusion_scale,
                attempt_points,
                attempt_dt,
            ),
            case_timeout_seconds,
        )

        if payload["status"] == "ok":
            return {
                "status": "ok",
                "diffusion_scale": float(diffusion_scale),
                "activity_label": activity_regime["label"],
                "activity_code": activity_regime["code"],
                "weight_index": weight_index,
                "sight_weight": sight_weight,
                "psi_value": payload["result"][3],
                "attempt_index": attempt_index,
                "attempt_points": attempt_points,
                "attempt_dt": attempt_dt,
            }

        if payload["status"] == "timeout":
            failure_messages.append(
                (
                    f"Timeout for {activity_regime['label']} ({activity_regime['code']}), "
                    f"D scale={diffusion_scale:g}, w={sight_weight:g}, attempt "
                    f"{attempt_index} with {attempt_points} points and dt={attempt_dt:g}."
                )
            )
        else:
            failure_messages.append(
                (
                    f"Failure for {activity_regime['label']} ({activity_regime['code']}), "
                    f"D scale={diffusion_scale:g}, w={sight_weight:g}, attempt "
                    f"{attempt_index} with {attempt_points} points and dt={attempt_dt:g}: "
                    f"{summarize_error_message(payload['message'])}"
                )
            )

    return {
        "status": "failed",
        "diffusion_scale": float(diffusion_scale),
        "activity_label": activity_regime["label"],
        "activity_code": activity_regime["code"],
        "weight_index": weight_index,
        "sight_weight": sight_weight,
        "psi_value": np.nan,
        "messages": failure_messages,
    }


def run_all_cases(
    lighting_regime,
    activity_regimes,
    sight_weights,
    diffusion_scales,
    number_of_points,
    dt,
    max_workers,
    case_timeout_seconds,
):
    psi_by_diffusion = {
        float(diffusion_scale): {
            activity_regime["label"]: [np.nan for _ in sight_weights]
            for activity_regime in activity_regimes
        }
        for diffusion_scale in diffusion_scales
    }
    case_specs = [
        (
            diffusion_index,
            regime_index,
            weight_index,
            diffusion_scale,
            activity_regime,
            sight_weight,
        )
        for diffusion_index, diffusion_scale in enumerate(diffusion_scales)
        for regime_index, activity_regime in enumerate(activity_regimes)
        for weight_index, sight_weight in enumerate(sight_weights)
    ]

    recovered_cases = []
    failed_cases = []

    def record_case_result(case_result):
        psi_by_diffusion[case_result["diffusion_scale"]][case_result["activity_label"]][
            case_result["weight_index"]
        ] = case_result["psi_value"]

        if case_result["status"] == "ok":
            if case_result["attempt_index"] > 1:
                recovered_cases.append(case_result)
                print(
                    (
                        f"Recovered {case_result['activity_label']} ({case_result['activity_code']}), "
                        f"D scale={case_result['diffusion_scale']:g}, "
                        f"w={case_result['sight_weight']:g} on attempt "
                        f"{case_result['attempt_index']} with {case_result['attempt_points']} "
                        f"points and dt={case_result['attempt_dt']:g}."
                    ),
                    flush=True,
                )
            else:
                print(
                    (
                        f"Finished {case_result['activity_label']} ({case_result['activity_code']}), "
                        f"D scale={case_result['diffusion_scale']:g}, "
                        f"w={case_result['sight_weight']:g}"
                    ),
                    flush=True,
                )
            return

        failed_cases.append(case_result)
        for message in case_result["messages"]:
            print(message, flush=True)
        print(
            (
                f"Skipped {case_result['activity_label']} ({case_result['activity_code']}), "
                f"D scale={case_result['diffusion_scale']:g}, "
                f"w={case_result['sight_weight']:g} after all retries."
            ),
            flush=True,
        )

    if max_workers <= 1:
        for case_spec in case_specs:
            record_case_result(
                run_case_with_retries(
                    case_spec,
                    lighting_regime,
                    number_of_points,
                    dt,
                    case_timeout_seconds,
                )
            )
        return psi_by_diffusion, recovered_cases, failed_cases

    with ThreadPoolExecutor(max_workers=min(max_workers, len(case_specs))) as executor:
        future_to_case = {
            executor.submit(
                run_case_with_retries,
                case_spec,
                lighting_regime,
                number_of_points,
                dt,
                case_timeout_seconds,
            ): case_spec
            for case_spec in case_specs
        }

        for future in as_completed(future_to_case):
            record_case_result(future.result())

    return psi_by_diffusion, recovered_cases, failed_cases


def format_diffusion_title(diffusion_scale):
    if np.isclose(diffusion_scale, 0.1):
        return r"$D/10$"
    if np.isclose(diffusion_scale, 1.0):
        return r"$D$"
    if np.isclose(diffusion_scale, 10.0):
        return r"$10D$"
    return rf"${diffusion_scale:g}D$"


def save_spread_plot(
    psi_by_diffusion,
    sight_weights,
    diffusion_scales,
    activity_regimes,
    t_sunset,
    output_path,
):
    figure, axes = plt.subplots(
        1,
        len(diffusion_scales),
        figsize=(5.2 * len(diffusion_scales), 5.1),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    legend_handles = []
    legend_labels = []

    for axis, diffusion_scale in zip(axes, diffusion_scales):
        for regime in activity_regimes:
            line, = axis.plot(
                sight_weights,
                psi_by_diffusion[float(diffusion_scale)][regime["label"]],
                color=regime["color"],
                marker=regime["marker"],
                linewidth=2.2,
                markersize=5.8,
                label=regime["label"],
            )
            if axis is axes[0]:
                legend_handles.append(line)
                legend_labels.append(f"{regime['label']} ({regime['code']})")

        axis.set_title(format_diffusion_title(diffusion_scale))
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(np.linspace(0.0, 1.0, 6))
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.set_xlabel("w")
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"$\Psi$")
    figure.suptitle(
        "Sleep-pattern spread across diffusion levels\n"
        f"$t_{{sunset}}={t_sunset:g}$"
    )
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
        title="Circadian activity pattern",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(figure)


def main():
    args = parse_args()
    sight_weights = tuple(float(weight) for weight in args.weights)
    diffusion_scales = tuple(float(scale) for scale in args.diffusion_scales)

    if any(weight < 0.0 or weight > 1.0 for weight in sight_weights):
        raise ValueError("All weights must lie in the interval [0, 1].")

    if not 0.0 <= args.t_sunset <= 1.0:
        raise ValueError("t_sunset must lie in the interval [0, 1].")

    if any(scale <= 0.0 for scale in diffusion_scales):
        raise ValueError("All diffusion scales must be positive.")

    rounded_scales = [round(scale, 12) for scale in diffusion_scales]
    if len(set(rounded_scales)) != len(rounded_scales):
        raise ValueError("diffusion scales must be distinct.")

    if args.number_of_points < 2:
        raise ValueError("number_of_points must be at least 2.")

    if args.dt <= 0.0:
        raise ValueError("dt must be positive.")

    if args.max_workers < 1:
        raise ValueError("max_workers must be at least 1.")

    if args.case_timeout_seconds <= 0.0:
        raise ValueError("case-timeout-seconds must be positive.")

    lighting_regime = build_lighting_regime(args.t_sunset, args.dt)
    output_path = args.output_path.resolve()
    psi_by_diffusion, recovered_cases, failed_cases = run_all_cases(
        lighting_regime,
        ACTIVITY_REGIMES,
        sight_weights,
        diffusion_scales,
        args.number_of_points,
        args.dt,
        args.max_workers,
        args.case_timeout_seconds,
    )
    save_spread_plot(
        psi_by_diffusion,
        sight_weights,
        diffusion_scales,
        ACTIVITY_REGIMES,
        args.t_sunset,
        output_path,
    )
    print(
        f"Recovered {len(recovered_cases)} case(s) with degraded retries; "
        f"skipped {len(failed_cases)} case(s)."
    )
    print(f"Saved spread plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())