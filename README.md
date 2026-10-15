# RecipeCF

Implementation of **RecipeCF: Support-Aware Multivariable Process Response Modeling for Operating-Point Decision Support**. The model predicts a scalar outcome, its action Jacobian, and a calibrated interval for a joint continuous action. A conditional action-density gate determines whether the query is admitted.

## Installation

Use Python 3.10 or later. A CPU is sufficient; this implementation keeps tensors on CPU.

```bash
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e .
```

## Data

The included generator provides independent process-style observational units, autoregressive Beta action assignment, a known nonlinear response surface, action gradients, and counterfactual evaluation grids:

```bash
recipecf generate --output data/process.npz --samples 24000 --conditions 30 --actions 4 --confounding 0.5 --noise 0.1
recipecf prepare --input data/process.npz --output data/process
```

`prepare` splits independent groups in sorted order into 70% training, 15% calibration, and 15% test. Use `--shuffle --seed 0` only when chronological order is not needed. Missing condition values are imputed with training medians; training means and standard deviations normalize conditions. Actions must already be normalized to the open unit interval, and outcomes retain their supplied units. The preprocessing parameters are written to `preprocessing.json`.

For external data, supply an NPZ archive with numeric arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| `x` | N × Dx | Conditions available before action assignment |
| `a` | N × Da | Joint action, strictly between zero and one |
| `y` | N | Post-action outcome; lower is preferred |
| `group` | N | Optional integer entity IDs; related rows stay together |
| `sign` | N × Da | Optional approved derivative priors: -1, 0, +1; zero disables a coordinate |
| `jacobian` | N × Da | Optional ground-truth derivative, used only in evaluation |
| `grid_a` | N × G × Da | Optional fixed counterfactual action grid |
| `grid_mu` | N × G | Known response at `grid_a`, used only in evaluation |

IHDP and News covariates can be converted to this schema after choosing a semi-synthetic action/outcome generator. Tennessee Eastman requires independently reset episodes and pre-action summaries. Supply actual simulator responses as `grid_mu`; observational outcomes alone do not define counterfactual ground truth.

## Training and inference

```bash
recipecf train --data data/process --output runs/full --variant full --epochs 200 --density-epochs 30 --seed 0
recipecf evaluate --model runs/full/model.pt --input data/process/test.npz --output runs/full/evaluation --oracle process
```

Training first fits marginal and conditional action flows. The response stage uses clipped stabilized density ratios, coverage-tempered local balance, a rank-8 condition/action interaction, heteroscedastic Gaussian loss, approved sign constraints, Jacobian smoothness, and finite-step consistency. Local balance detaches the condition representation, so its gradient adjusts assignment weights rather than discarding prognostic information.

The calibration partition supplies weighted normalized residuals and the support threshold. Selection uses the smallest candidate threshold giving at least 90% empirical weighted calibration coverage with at least 80% retention. If none satisfies the rule, training reports that condition instead of silently substituting a threshold. Query-specific weighted quantiles include the query mass at infinity. An infinite interval is retained as infinite in predictions and reported separately in metrics.

`model.pt` stores parameters, residuals, clipping bounds, threshold, and dimensions. `history.json` records the actual training objective. Evaluation writes `metrics.json`, `predictions.npz`, and `retention.json`. Predictions contain response, interval radius, Jacobian, support score, admission mask, and projected-gradient actions. Treat rows with `admitted=False` or `search_admitted=False` as abstentions.

Key parameters:

| Option | Default | Meaning |
| --- | --- | --- |
| `--width` | 128 | Hidden and condition representation width |
| `--rank` | 8 | Interaction rank |
| `--batch-size` | 512 | Minibatch size; also affects local effective sample size |
| `--lr` | 0.001 | Adam learning rate |
| `--anchors` | 32 | Training action anchors |
| `--bandwidth` | 0.2 | Gaussian action-neighborhood bandwidth |
| `--balance` | 1.0 | Local moment-loss coefficient inside assignment loss |
| `--restarts` | 8 | Evaluation action-search restarts |
| `--search-steps` | 100 | Projected-gradient steps per restart |
| `--search-lr` | 0.01 | Action-search step size |

The assignment coefficient is 0.20, derivative coefficient 0.10, smoothness coefficient 0.50, remainder coefficient 0.25, and rank coefficient 1e-4. Coverage tempering uses effective sample size 128 and temperature 24. These values are defined in `recipecf/cli.py` and `recipecf/model.py`.

## Baseline and ablations

```bash
recipecf train --data data/process --output runs/mlp --variant mlp --seed 0
recipecf evaluate --model runs/mlp/model.pt --input data/process/test.npz --output runs/mlp/evaluation
recipecf train --data data/process --output runs/no_balance --variant no_balance --seed 0
recipecf evaluate --model runs/no_balance/model.pt --input data/process/test.npz --output runs/no_balance/evaluation
```

Repeat with `fixed_balance`, `no_jacobian`, `no_diff`, `no_gate`, `concat`, or `full_rank`. The MLP baseline uses the same condition representation and action basis with an ordinary concatenation head, factual likelihood, and no density weighting, derivative regularization, or inference gate. Other variants remove or replace the corresponding RecipeCF component. `full_rank` expands the interaction rank to the representation width.

Vary `generate --confounding`, `--actions`, `--noise`, and `--sign-corruption` to construct the corresponding stress conditions. Reuse the same seeds across methods. Each generated dataset must be prepared separately. For repeated-seed analysis, give each run its own output directory.

## Metrics and figures

```bash
recipecf plot --input runs --output runs/figures
recipecf compare --first runs/full/evaluation/predictions.npz --second runs/mlp/evaluation/predictions.npz --output runs/comparison.json --resamples 10000
```

The plot command exports `comparison.csv` and one bar chart per shared numeric metric. The paired comparison recomputes normalized MISE on the intersection of both models' supported grids for every entity. It writes the mean paired difference, percentile 95% bootstrap interval, and a centered-bootstrap two-sided p value. Negative differences favor the first model.

Apply Holm correction to a declared family of benchmark comparisons after producing one comparison JSON per benchmark:

```bash
recipecf holm --inputs runs/comparison.json runs/other_benchmark/comparison.json --output runs/holm.json
```

| Metric | Definition |
| --- | --- |
| `normalized_mise` | Per-entity supported-grid squared error divided by supported response variance, then averaged |
| `supported_grid_regret` | True response of the predicted grid minimizer minus the true supported-grid minimum |
| `supported_search_regret` | Oracle response of the projected-gradient decision minus an oracle-search reference |
| `factual_rmse` | RMSE on all factual test observations |
| `gradient_sign_accuracy` | Sign agreement on admitted coordinates with nonzero true derivative |
| `jacobian_magnitude_error` | Absolute difference of predicted and true Jacobian Euclidean norms |
| `local_balance_error` | Unweighted mean of squared local representation/action cross-moments with stabilized weights |
| `interval_coverage`, `interval_width` | Admitted factual coverage and average interval width |
| `abstention_rate` | Fraction of factual queries rejected by support |
| `parameters`, `train_seconds`, `batch_one_latency_ms` | Model size and locally measured computational costs |

Counterfactual metrics require `grid_a` and `grid_mu`; derivative metrics require `jacobian`. Missing ground truth does not produce substitute scores. Risk/retention curves use fixed predictions and change only the admitted population.

## Implementation conventions

The manuscript specifies response interactions and assignment equations but does not supply the complete benchmark generators. The included process generator defines a quadratic condition-dependent response with adjacent pair interactions and an all-action product; its formula is in `surface`. It is a concrete benchmark convention. The generated grids use uniform random actions and remain fixed across methods.

Action densities use six logit-space affine coupling layers. Cubic B-splines use fixed internal knots at 0.25, 0.50, and 0.75. Training has a fixed epoch budget rather than selecting epochs on the calibration data. The implemented comparison baseline is MLP-Concat; DRNet, VCNet, ACFR, DTRNet, DCNet, and WDBN are separate methods and are not aliases for that baseline.

Grid regret evaluates candidate ranking on supplied ground truth. `evaluate --oracle process` also computes continuous decision regret using the included response function and eight-restart, 100-step oracle search. This is a numerical oracle-search reference, so search error can yield a negative estimated regret. For external simulators, `--oracle path/to/oracle.py` loads a function `response(raw_x, actions)` returning a differentiable PyTorch vector. `raw_x` contains the imputed conditions before standardization; the oracle must use the same response units as `y`. Computational costs describe the machine executing the command.
