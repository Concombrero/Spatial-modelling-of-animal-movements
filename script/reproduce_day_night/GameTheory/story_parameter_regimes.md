# Story Parameter Regimes

This note records the strongest parameter regimes found with the current
`script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh` workflow.

All bash runs below should use `--python .venv/bin/python` in this workspace,
because the default `python` executable is missing `nashpy`.

## Story 1: closest validated match

Status: completed bash pipeline run under
`script/reproduce_day_night/GameTheory/output/Test2`.

Parameters:

1. `w1 = 0.3`
2. `w2 = 1.0`
3. `t_sunset = 0.5`
4. `prey sight radius = 0.06`
5. `prey smell radius = 0.18`
6. `predator sight radius = 0.18`
7. `predator smell radius = 0.06`
8. `chi12 = -0.35`
9. `chi21 = 0.35`
10. `chi22 = 0.0`

Pipeline command:

```bash
script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh \
  --output-dir script/reproduce_day_night/GameTheory/output/Test2 \
  --python .venv/bin/python \
  --mean-x-axes cycle1,cycle2 \
  --replicator-time-span 2000 \
  --replicator-time-steps 400000 \
  --weights 0.3 1 \
  --prey-sight-radius 0.06 \
  --predator-sight-radius 0.18 \
  --prey-smell-radius 0.18 \
  --predator-smell-radius 0.06 \
  --chi22 0 \
  --chi21 0.35 \
  --chi12 -0.35
```

Observed replicator outcome from the saved payoff matrix:

1. Prey final dominant regime: `N` with final share `0.5484`.
2. Prey second regime: `M2` with final share `0.4179`.
3. Predator transient diurnal peak: `D` reaches `0.1744` at `t ~= 2.85`.
4. Predator final dominant regime: `M2` with final share `0.8070`.
5. Predator second regime: `P2` with final share `0.1930`.

Interpretation:

- This is the best validated Story 1 approximation found in the current tree.
- It captures nocturnal prey persistence and a brief early diurnal predator
  phase.
- It does not reproduce a strong sustained diurnal predator takeover, so it
  should be treated as the closest tractable match rather than an exact hit.

## Story 2: strongest short-day visual candidate

Status: identified and launched through the bash pipeline, but not fully
validated by a completed payoff-matrix run inside the interactive session
budget.

Parameters to continue from:

1. `w1 = 1.0`
2. `w2 = 1.0`
3. `t_sunset = 0.3`
4. `prey sight radius = 0.10`
5. `prey smell radius = 0.20`
6. `predator sight radius = 0.10`
7. `predator smell radius = 0.20`
8. `strategy subset = D, N, P1, M1`

Rationale:

1. Short days penalize shifted daylight strategies and favor dawn-touching
   fragmented regimes.
2. High visual reliance should strengthen the refuge advantage of `P1` and
   `M1`.
3. This is the most defensible Story 2 candidate consistent with the article
   intuition and the current search evidence.

Fast validation command used in-session:

```bash
bash script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh \
  --output-dir script/reproduce_day_night/GameTheory/output/story2_minimal \
  --python .venv/bin/python \
  --mean-x-axes cycle1,cycle2 \
  --strategy-codes D,N,P1,M1 \
  --t-sunset 0.3 \
  --weights 1.0 1.0 \
  --number-of-points 4 \
  --dt 1.0 \
  --number-of-cycles 1 \
  --observation-window 1.0 \
  --replicator-time-span 60 \
  --replicator-time-steps 600 \
  --replicator-plot-style line \
  --max-workers 4
```

Higher-fidelity command that was also launched but did not finish in-session:

```bash
bash script/reproduce_day_night/GameTheory/run_payoff_pipeline.sh \
  --output-dir script/reproduce_day_night/GameTheory/output/story2_candidate \
  --python .venv/bin/python \
  --mean-x-axes cycle1,cycle2 \
  --t-sunset 0.3 \
  --weights 1.0 1.0 \
  --number-of-points 24 \
  --dt 0.2 \
  --number-of-cycles 2 \
  --observation-window 1.0 \
  --replicator-time-span 200 \
  --replicator-time-steps 4000 \
  --replicator-plot-style line \
  --max-workers 4
```

Current interpretation:

- This is the best Story 2 candidate found so far.
- It should still be considered a candidate, not a validated match, until one
  of the short-day bash runs completes and the replicator output confirms that
  `P1` or `M1` dominates while `D` and `N` recede.

## Ruled-out historical dynamic-game reference

The committed historical evolutionary-game run from commit
`69de73f1481d7330c758736a6cd44c4a340de574` does not reproduce either story.

Parameters:

1. `w1 = 0.25`
2. `w2 = 0.75`
3. `t_sunset = 0.5`
4. `shared sight radius = 0.10`
5. `shared smell radius = 0.20`

Observed long-run behavior:

1. Prey converges mainly to `D`.
2. Predator converges mainly to `P1`, with a secondary `N` share.

This historical run is useful as a reference, but it should not be reported as
either a Story 1 or Story 2 solution.