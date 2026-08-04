from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SOLVER_DIRECTORY = PACKAGE_ROOT / "Solver"
BASIC_SIMULATION_DIRECTORY = PACKAGE_ROOT / "BasicSimulation"
GAME_THEORY_DIRECTORY = PACKAGE_ROOT / "GameTheory"

SOLVER_OUTPUT_DIRECTORY = SOLVER_DIRECTORY / "output"
BASIC_SIMULATION_OUTPUT_DIRECTORY = BASIC_SIMULATION_DIRECTORY / "output"
GAME_THEORY_OUTPUT_DIRECTORY = GAME_THEORY_DIRECTORY / "output"
GAME_THEORY_PAYOFF_OUTPUT_DIRECTORY = GAME_THEORY_OUTPUT_DIRECTORY / "payoff"
GAME_THEORY_STORY_SEARCH_OUTPUT_DIRECTORY = (
    GAME_THEORY_OUTPUT_DIRECTORY / "story_search"
)


def ensure_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def solver_output_path(*parts):
    return SOLVER_OUTPUT_DIRECTORY.joinpath(*parts)


def basic_simulation_output_path(*parts):
    return BASIC_SIMULATION_OUTPUT_DIRECTORY.joinpath(*parts)


def game_theory_output_path(*parts):
    return GAME_THEORY_OUTPUT_DIRECTORY.joinpath(*parts)


def game_theory_payoff_output_path(*parts):
    return GAME_THEORY_PAYOFF_OUTPUT_DIRECTORY.joinpath(*parts)


def story_search_output_path(*parts):
    return GAME_THEORY_STORY_SEARCH_OUTPUT_DIRECTORY.joinpath(*parts)