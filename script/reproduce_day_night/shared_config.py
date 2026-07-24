"""Shared visual style and simulation defaults for day-night experiments."""

from __future__ import annotations

import copy
import math
import os


PLOT_STYLE = {
    "day_color": "#F2C14E",
    "night_color": "#3A86FF",
    "inactive_face_color": "#EEF4FB",
    "inactive_edge_color": "#7C97B6",
    "axis_color": "#111111",
    "guide_color": "#C5CCD5",
    "consensus_color": "#BDBDBD",
    "axes_title_fontsize": 16,
    "figure_title_fontsize": 18,
    "legend_fontsize": 13,
    "legend_title_fontsize": 14,
}

ACTIVITY_REGIMES = (
    {
        "code": "D",
        "label": "Diurnal",
        "group": "Diurnal",
        "periods": ((0.0, 0.5),),
        "color": PLOT_STYLE["day_color"],
        "marker": "o",
    },
    {
        "code": "N",
        "label": "Nocturnal",
        "group": "Nocturnal",
        "periods": ((0.5, 1.0),),
        "color": PLOT_STYLE["night_color"],
        "marker": "s",
    },
    {
        "code": "P1",
        "label": "Polyphasic 1",
        "group": "Polyphasic",
        "periods": ((0.0, 0.25), (0.5, 0.75)),
        "color": "#2A9D8F",
        "marker": "^",
    },
    {
        "code": "P2",
        "label": "Polyphasic 2",
        "group": "Polyphasic",
        "periods": ((0.25, 0.5), (0.75, 1.0)),
        "color": "#8AC926",
        "marker": "D",
    },
    {
        "code": "M1",
        "label": "Matutinal 1",
        "group": "Matutinal",
        "periods": ((0.0, 0.25), (0.75, 1.0)),
        "color": "#FF7F51",
        "marker": "v",
    },
    {
        "code": "M2",
        "label": "Matutinal 2",
        "group": "Matutinal",
        "periods": ((0.25, 0.75),),
        "color": "#D1495B",
        "marker": "P",
    },
)

ACTIVITY_REGIME_BY_CODE = {
    regime["code"]: regime for regime in ACTIVITY_REGIMES
}
ACTIVITY_CODES = tuple(regime["code"] for regime in ACTIVITY_REGIMES)
ACTIVITY_LABELS = {
    regime["code"]: regime["label"] for regime in ACTIVITY_REGIMES
}
ACTIVITY_COLORS = {
    regime["code"]: regime["color"] for regime in ACTIVITY_REGIMES
}
ACTIVITY_MARKERS = {
    regime["code"]: regime["marker"] for regime in ACTIVITY_REGIMES
}

DIURNAL_NOCTURNAL_CODES = ("D", "N")
POLYPHASIC_CODES = ("P1", "P2")
MATUTINAL_CODES = ("M1", "M2")

ONE_POPULATION_SIMULATION_CONFIG = {
    "base": {
        "number_of_populations": 1,
        "number_of_points": 256,
        "cycle_period": 1.0,
        "dt": 0.01,
        "day_start": 0.0,
        "sight_radius": 0.1,
        "smell_radius": 0.2,
        "coefficient_attraction": ((0.1,),),
        "coefficient_diffusion": (0.04,),
        "observation_window": 1.0,
    },
    "experiments": {
        "activity_const_heatmaps": {
            "number_of_cycles": 3,
            "weights": (0.0, 0.5, 1.0),
            "sunset_values": (1.0, 0.5, 0.0),
        },
        "activity_const_spread": {
            "number_of_cycles": 3,
            "weights": tuple(round(0.1 * index, 1) for index in range(11)),
            "sunset_values": (1.0, 0.5, 0.0),
            "max_workers": 3,
        },
        "sleep_pattern_heatmaps": {
            "number_of_cycles": 2,
            "weights": (0.0, 0.5, 1.0),
            "t_sunset": 0.5,
            "activity_codes": ACTIVITY_CODES,
            "max_workers": min(6, os.cpu_count() or 1),
        },
        "sleep_pattern_spread": {
            "number_of_cycles": 2,
            "weights": tuple(round(0.1 * index, 1) for index in range(11)),
            "sunset_values": (0.0, 0.25, 0.5, 0.75, 1.0),
            "activity_codes": ACTIVITY_CODES,
            "max_workers": min(16, os.cpu_count() or 1),
        },
        "spread_diurnal_vs_nocturnal": {
            "number_of_cycles": 2,
            "weights": tuple(round(0.1 * index, 1) for index in range(11)),
            "t_sunset": 0.5,
            "activity_codes": DIURNAL_NOCTURNAL_CODES,
            "max_workers": min(16, os.cpu_count() or 1),
        },
        "spread_polyphasic_matutinal": {
            "number_of_cycles": 2,
            "weights": tuple(round(0.1 * index, 1) for index in range(11)),
            "sunset_values": (0.3, 0.5, 0.7),
            "activity_codes": POLYPHASIC_CODES + MATUTINAL_CODES,
            "max_workers": min(16, os.cpu_count() or 1),
        },
        "spread_sleep_pattern_diffusion": {
            "number_of_cycles": 2,
            "weights": tuple(round(0.1 * index, 1) for index in range(11)),
            "t_sunset": 0.5,
            "activity_codes": ACTIVITY_CODES,
            "diffusion_scales": (0.1, 1.0, 10.0),
            "max_workers": min(6, os.cpu_count() or 1),
            "case_timeout_seconds": 300.0,
            "retry_point_scales": (1.0, 0.5, 0.25),
            "retry_dt_scales": (1.0, 1.0, 2.0),
            "min_retry_number_of_points": 64,
        },
    },
}

TWO_POPULATION_SIMULATION_CONFIG = {
    "base": {
        "number_of_populations": 2,
        "number_of_points": 128,
        "number_of_cycles": 4,
        "cycle_period": 1.0,
        "dt": 0.1,
        "day_start": 0.0,
        "t_sunset": 0.5,
        "weights": (0.5, 0.5),
        "sight_radius": 0.1,
        "smell_radius": 0.1,
        "initial_centers": (0.25, 0.70),
        "initial_width": 0.1,
        "diffusion": (0.04, 0.04),
        "attraction": (
            (0.1, -0.2),
            (0.2, 0.1),
        ),
        "reaction_rates": {
            "prey_growth": 0.1,
            "predator_decay": 0.04,
            "predation_rate": 0.25,
            "conversion_rate": 0.15,
        },
        "observation_window": 2.0,
        "payoff_mode": "population-integral",
    },
    "analysis": {
        "max_workers": min(16, os.cpu_count() or 1),
        "weight_sweep_values": tuple(round(0.1 * index, 10) for index in range(11)),
        "weight_sweep_payoff_mode": "population-integral",
        "replicator_time_span": 40.0,
        "replicator_time_steps": 800,
        "mean_x_axes": ("w1", "w2", "cycle1", "cycle2"),
        "mean_show_variance": False,
        "strategy_codes": ACTIVITY_CODES,
    },
}


def resolve_experiment_config(study_config, experiment_name):
    merged = copy.deepcopy(study_config["base"])
    merged.update(copy.deepcopy(study_config["experiments"][experiment_name]))
    return merged


def activity_regimes_for_codes(codes=None):
    if codes is None:
        return ACTIVITY_REGIMES

    return tuple(ACTIVITY_REGIME_BY_CODE[code] for code in codes)


def describe_lighting_regime(t_sunset):
    if math.isclose(t_sunset, 1.0):
        return "full day"
    if math.isclose(t_sunset, 0.0):
        return "full night"
    if math.isclose(t_sunset, 0.5):
        return "half day / half night"
    if t_sunset < 0.5:
        return "short day"
    return "long day"


def build_constant_activity_lighting_regime(
    t_sunset,
    *,
    total_time,
    cycle_period,
):
    t_sunset = float(t_sunset)
    if t_sunset < 0.0 or t_sunset > 1.0:
        raise ValueError("t_sunset must lie in the interval [0, 1].")

    long_cycle_period = float(total_time) + float(cycle_period)
    if math.isclose(t_sunset, 1.0):
        return {
            "label": "full day",
            "display_sunset": 1.0,
            "cycle_period": long_cycle_period,
            "day_start": 0.0,
            "day_end": float(total_time) + 0.5 * float(cycle_period),
            "show_transition_markers": False,
        }

    if math.isclose(t_sunset, 0.0):
        return {
            "label": "full night",
            "display_sunset": 0.0,
            "cycle_period": long_cycle_period,
            "day_start": float(total_time) + 0.25 * float(cycle_period),
            "day_end": float(total_time) + 0.75 * float(cycle_period),
            "show_transition_markers": False,
        }

    return {
        "label": describe_lighting_regime(t_sunset),
        "display_sunset": t_sunset,
        "cycle_period": float(cycle_period),
        "day_start": 0.0,
        "day_end": t_sunset * float(cycle_period),
        "show_transition_markers": True,
    }


def build_constant_activity_lighting_regimes(
    sunset_values,
    *,
    total_time,
    cycle_period,
):
    return tuple(
        build_constant_activity_lighting_regime(
            t_sunset,
            total_time=total_time,
            cycle_period=cycle_period,
        )
        for t_sunset in sunset_values
    )


def build_periodic_lighting_regime(
    t_sunset,
    *,
    dt,
    cycle_period,
    day_start,
):
    t_sunset = float(t_sunset)
    if t_sunset < 0.0 or t_sunset > 1.0:
        raise ValueError("t_sunset must lie in the interval [0, 1].")

    epsilon = min(max(0.5 * float(dt), 1.0e-6), 0.25 * float(cycle_period))
    effective_sunset = min(max(t_sunset, epsilon), 1.0 - epsilon)
    return {
        "label": describe_lighting_regime(t_sunset),
        "display_sunset": t_sunset,
        "cycle_period": float(cycle_period),
        "day_start": float(day_start),
        "day_end": effective_sunset * float(cycle_period),
    }


def build_periodic_lighting_regimes(
    sunset_values,
    *,
    dt,
    cycle_period,
    day_start,
):
    return tuple(
        build_periodic_lighting_regime(
            t_sunset,
            dt=dt,
            cycle_period=cycle_period,
            day_start=day_start,
        )
        for t_sunset in sunset_values
    )


def apply_plot_typography():
    import matplotlib

    matplotlib.rcParams.update(
        {
            "axes.titlesize": PLOT_STYLE["axes_title_fontsize"],
            "figure.titlesize": PLOT_STYLE["figure_title_fontsize"],
            "legend.fontsize": PLOT_STYLE["legend_fontsize"],
            "legend.title_fontsize": PLOT_STYLE["legend_title_fontsize"],
        }
    )


__all__ = [
    "ACTIVITY_CODES",
    "ACTIVITY_COLORS",
    "ACTIVITY_LABELS",
    "ACTIVITY_MARKERS",
    "ACTIVITY_REGIME_BY_CODE",
    "ACTIVITY_REGIMES",
    "DIURNAL_NOCTURNAL_CODES",
    "MATUTINAL_CODES",
    "ONE_POPULATION_SIMULATION_CONFIG",
    "PLOT_STYLE",
    "POLYPHASIC_CODES",
    "TWO_POPULATION_SIMULATION_CONFIG",
    "activity_regimes_for_codes",
    "apply_plot_typography",
    "build_constant_activity_lighting_regime",
    "build_constant_activity_lighting_regimes",
    "build_periodic_lighting_regime",
    "build_periodic_lighting_regimes",
    "describe_lighting_regime",
    "resolve_experiment_config",
]