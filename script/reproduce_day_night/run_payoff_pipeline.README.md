# Payoff Pipeline Runner

`run_payoff_pipeline.sh` exécute en une seule commande la chaîne complète suivante dans un dossier de sortie choisi:

1. calcul de la matrice de payoff avec `payoff_matrix.py`
2. analyse des moyennes avec `payoff_mean_analysis.py`
3. analyse minmax/maxmin avec `payoff_minmax_maxmin.py`
4. calcul des équilibres de Nash avec `payoff_nash_equilibrium.py`
5. dynamique du replicateur avec `payoff_replicator_analysis.py`

Le script écrit aussi un fichier `pipeline_command.sh` pour rejouer exactement la même commande, ainsi qu'un dossier `logs/` avec un log par étape.

## Emplacement

Script:

- `script/reproduce_day_night/run_payoff_pipeline.sh`

## Usage minimal

Depuis la racine du dépôt:

```bash
bash script/reproduce_day_night/run_payoff_pipeline.sh \
  --output-dir script/reproduce_day_night/output/full_pipeline_run
```

## Exemples

Réduire la durée du jour et diminuer le poids visuel des proies:

```bash
bash script/reproduce_day_night/run_payoff_pipeline.sh \
  --output-dir script/reproduce_day_night/output/short_day_run \
  --t-sunset 0.35 \
  --weights 0.2 0.7
```

Rendre le prédateur plus efficace visuellement et produire un replicator plus long:

```bash
bash script/reproduce_day_night/run_payoff_pipeline.sh \
  --output-dir script/reproduce_day_night/output/predator_day_advantage \
  --predator-sight-radius 0.18 \
  --prey-sight-radius 0.06 \
  --weights 0.3 0.9 \
  --replicator-time-span 120 \
  --replicator-time-steps 2400
```

Ne produire que certaines analyses moyennes:

```bash
bash script/reproduce_day_night/run_payoff_pipeline.sh \
  --output-dir script/reproduce_day_night/output/cycle_focus \
  --mean-x-axes cycle1,cycle2 \
  --mean-show-variance true
```

Utiliser un interpréteur Python spécifique:

```bash
bash script/reproduce_day_night/run_payoff_pipeline.sh \
  --output-dir script/reproduce_day_night/output/venv_run \
  --python .venv/bin/python
```

## Structure des sorties

Le dossier passé à `--output-dir` contient normalement:

- `payoff_matrix.csv`
- `case_payoffs.csv`
- `run_config.json`
- `payoff_matrix.png`
- `payoff_minmax_maxmin.json`
- `payoff_nash_equilibrium.json`
- `mean_analysis/mean_vs_*.png`
- `replicator_analysis/prey_strategy_frequencies.png`
- `replicator_analysis/predator_strategy_frequencies.png`
- `population_heatmaps/` si tu demandes les heatmaps ou la sortie complète
- `logs/*.log`
- `pipeline_command.sh`

## Paramètres configurables

### Options générales

- `--output-dir DIR`
  Dossier de sortie principal. Obligatoire.
- `--python BIN`
  Exécutable Python à utiliser. Par défaut: `python`.
- `--mean-x-axes LIST`
  Liste séparée par des virgules pour l'analyse mean. Valeurs utiles: `w1`, `w2`, `cycle1`, `cycle2`.
  Par défaut: `w1,w2,cycle1,cycle2`.
- `--mean-show-variance BOOL`
  `true` ou `false`. Ajoute le sous-graphe de variance dans l'analyse mean.
- `--replicator-time-span FLOAT`
  Temps final de la dynamique du replicateur.
- `--replicator-time-steps INT`
  Nombre de points de sortie pour les courbes du replicateur.
- `--replicator-plot-style STYLE`
  `line` ou `stacked`.

### Paramètres jour/nuit et perception

Ce sont les leviers les plus directs pour faire bouger la préférence diurne/nocturne.

- `--t-sunset FLOAT`
  Proportion de jour dans un cycle, dans `[0, 1]`.
- `--weights W1 W2`
  Poids de la vue pour les proies puis les prédateurs.
- `--sight-radius FLOAT`
  Rayon de vue commun aux deux populations.
- `--prey-sight-radius FLOAT`
  Rayon de vue des proies.
- `--predator-sight-radius FLOAT`
  Rayon de vue des prédateurs.
- `--smell-radius FLOAT`
  Rayon d'odorat commun aux deux populations.
- `--prey-smell-radius FLOAT`
  Rayon d'odorat des proies.
- `--predator-smell-radius FLOAT`
  Rayon d'odorat des prédateurs.

Interprétation pratique:

- diminuer `t-sunset` raccourcit la journée et peut affaiblir `D`
- diminuer `w1` ou `prey-sight-radius` réduit l'avantage visuel des proies
- augmenter `w2` ou `predator-sight-radius` renforce la pression des prédateurs pendant le jour
- augmenter la part olfactive lisse l'asymétrie jour/nuit

### Paramètres de discrétisation et horizon PDE

- `--number-of-points INT`
  Nombre de points spatiaux.
- `--dt FLOAT`
  Pas de temps stocké.
- `--number-of-cycles INT`
  Nombre de cycles simulés.
- `--observation-window FLOAT`
  Fenêtre finale utilisée pour intégrer le payoff d'overlap.

### Paramètres de réaction Lotka-Volterra

- `--prey-growth FLOAT`
  Taux de croissance des proies.
- `--predator-decay FLOAT`
  Taux de décroissance des prédateurs.
- `--predation-rate FLOAT`
  Intensité de prédation.
- `--conversion-rate FLOAT`
  Conversion proie -> prédateur.

Ces paramètres changent la pression écologique globale et peuvent rendre l'évitement temporel plus ou moins avantageux.

### Paramètres d'attraction et diffusion

- `--chi11 FLOAT`
  Auto-attraction des proies.
- `--chi12 FLOAT`
  Réponse des proies aux prédateurs.
- `--chi21 FLOAT`
  Réponse des prédateurs aux proies.
- `--chi22 FLOAT`
  Auto-attraction des prédateurs.
- `--diffusion D1 D2`
  Diffusions des proies et des prédateurs.

Ces paramètres modifient la séparation ou le regroupement spatial, donc directement l'overlap utilisé comme payoff.

### Conditions initiales

- `--initial-centers X1 X2`
  Centres initiaux des gaussiennes proies/prédateurs.
- `--initial-width FLOAT`
  Largeur initiale commune.

### Sorties avancées et calcul parallèle

- `--heatmap-prey CODE`
- `--heatmap-predator CODE`
  À fournir ensemble si tu veux restreindre les heatmaps à une seule paire de stratégies parmi `D`, `N`, `P1`, `P2`, `M1`, `M2`.
- `--max-workers INT`
  Nombre de workers pour le calcul de la matrice de payoff.

## Remarques

- Le script suppose que l'environnement Python choisi contient au moins `numpy`, `matplotlib`, `scipy` et `nashpy`.
- Les analyses `nash` et `replicator` dépendent de `nashpy`.
- Si tu lances la pipeline sur un seul run, les courbes mean en `w1` et `w2` auront souvent un seul point. Les axes `cycle1` et `cycle2` sont souvent plus informatifs dans ce cas.