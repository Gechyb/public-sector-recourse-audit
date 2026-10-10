# public-sector-recourse-audit

An audit of **algorithmic recourse** in a public-sector-style eligibility model:
when a model denies someone, is there a realistic change they could make to get
approved, and **is that path equally available and equally costly across groups**?

Explainability work usually asks *why* a model made a decision. Recourse asks the
question that matters to the person affected: *what can I do about it?* A model can
have similar error rates across groups and still give some groups a far harder
(or impossible) path to a favourable decision. This project measures that gap.

## Setting

- **Data:** UCI Adult (1994 US Census extract, 48,842 people), loaded from OpenML
  and cached to `data/raw/`. It stands in for census-style administrative data used
  in eligibility screening. The favourable outcome ("approved") is `income > 50K`.
  If OpenML is unreachable, the loader falls back to a schema-compatible synthetic
  dataset with deliberately injected disparities. Only use that fallback for testing.
- **Modern replication:** ACS 2023 North Carolina (56,486 workers), downloaded with
  [folktables](https://github.com/socialfoundations/folktables) using the ACSIncome
  definition (Ding et al. 2021). It is recoded to Adult's column names, and the same
  recourse rules apply wherever a feature exists
  ([`config/constraints_acs_nc.yaml`](config/constraints_acs_nc.yaml)). This asks
  whether gaps found on the 1994 benchmark still hold in current data. "Approved"
  means income above **$103K**, Adult's 1994 $50K in 2023 dollars, so both datasets
  use the same real-dollar bar. The ACSIncome standard of $50K (not inflation-adjusted)
  runs as a robustness check ([`config/constraints_acs_nc_50k.yaml`](config/constraints_acs_nc_50k.yaml)).
- **Models:** logistic regression and gradient boosting (`HistGradientBoostingClassifier`).
  Sex and race are **excluded** from model inputs by default
  (`use_protected_in_model: false`), as most public agencies are required to do.
  Proxies such as `relationship` and `marital_status` remain.
- **Audit groups:** sex, race (White / Black / Other), age group (<30, 30-49, 50+).

## Recourse definition

Everything that defines recourse lives in [`config/constraints.yaml`](config/constraints.yaml):

| Feature | Type | Allowed change | Cost |
|---|---|---|---|
| `education_num` | numeric | increase only, up to 16 | 1.0 per year |
| `hours_per_week` | numeric | any direction, 1-80, steps of 5 | 0.05 per hour (20 h = 1) |
| `workclass` | categorical | switch to any listed sector | 1.0 |
| `occupation` | categorical | switch to any listed occupation | 1.5 |
| age, marital status, relationship, native country, capital gains/losses | immutable | none | none |

An applicant **has recourse** if some combination of up to 3 allowed actions, with
total cost ≤ `max_cost` (6.0), moves the model's approval probability to ≥ 0.5.
Costs are abstract effort units. Use them to compare groups, not as absolute measures.

### Search algorithm ([`src/recourse.py`](src/recourse.py))

This search is model-agnostic. It only calls `predict_proba` on raw DataFrames.

1. Try every single-feature action.
2. Try `n_candidates` random multi-feature actions.
3. Take the cheapest approving action and **greedily shrink** it toward the
   original values. This drops unnecessary changes and reduces cost while
   approval still holds.

The search is heuristic, not exact. "No recourse found" means none was found
within the action space and budget. It does not prove that none exists.

## Metrics ([`src/metrics.py`](src/metrics.py))

- **Model performance by group:** approval rate, accuracy, TPR, FPR, and
  **false-negative rate** (qualified people who are denied).
- **Recourse rate:** share of denied applicants with feasible recourse.
- **Recourse cost:** mean, median, and p90 cost among those with recourse, plus
  the number of features changed.
- **Disparities:** within-attribute gap (max - min) and ratio (max / min), naming the
  worst-off group.
- **Action mix:** how often each actionable feature appears in a group's recourse.

## Project layout

```
public-sector-recourse-audit/
├── README.md
├── requirements.txt
├── LICENSE
├── config/constraints.yaml     # data, model, action space, search settings
├── src/
│   ├── data_loader.py          # load Adult / synthetic, derive groups, split X / y / groups
│   ├── model.py                # sklearn pipelines (logistic, gbm), train / evaluate
│   ├── recourse.py             # ActionSpace + RecourseSearch
│   ├── metrics.py              # group performance, recourse disparities
│   └── plots.py                # audit figures
├── run_audit.py                # end-to-end pipeline
├── data/processed/             # result CSVs (+ trained models, git-ignored)
├── figures/                    # PNG figures
└── app/app.py                  # optional Streamlit explorer
```

## Usage

Requires Python 3.10+ and `make`. After cloning:

```bash
make setup    # create .venv, install requirements, download the data
make test     # automated correctness tests
make lint     # code style check (flake8)
make quick    # smoke test on every config (~1 min)
make audit    # full audit (a few minutes)
make app      # interactive Streamlit explorer
make help     # list all targets
```

Or manually, without `make`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_audit.py                    # full audit (a few minutes)
python run_audit.py --quick            # smoke test (~15 s)
python run_audit.py --models logistic --sample-size 2000

streamlit run app/app.py               # optional interactive explorer
```

To change what counts as recourse (e.g. make occupation immutable, or raise the
budget), edit `config/constraints.yaml` and re-run.

## Outputs

For each dataset (`adult`, `acs_nc_2023`) and model `{m}`, in `data/processed/<dataset>/`:

| File | Contents |
|---|---|
| `model_summary.csv` | accuracy, AUC, approval rate, overall recourse rate and cost, unconstrained recourse and violation rates |
| `group_performance_{m}.csv` | approval / error rates per group |
| `recourse_individual_{m}.csv` | one row per audited denied applicant, with the recommended changes |
| `recourse_by_group_{m}.csv` | recourse rate and cost per group |
| `recourse_disparities_{m}.csv` | gaps and ratios between best- and worst-off groups |
| `feature_changes_{m}.csv` | share of recourse actions using each feature, per group |
| `recourse_unconstrained_{variant}_{m}.csv` | unconstrained baseline per applicant, with the constraints each recommendation breaks |
| `violations_by_group_{m}.csv` | unconstrained recourse rate and violation rate per group, per variant |
| `violation_types_{m}.csv` | how often each constraint (e.g. `immutable:marital_status`) is broken |

The matching figures in `figures/<dataset>/` are `recourse_rate_{m}.png`,
`recourse_mean_cost_{m}.png`, `recourse_cost_dist_{m}.png`,
`feature_changes_{m}.png`, `false_negative_rate_{m}.png`, `violation_rate_{m}.png`,
and `violation_types_{m}.png`.

The unconstrained baseline (config section `unconstrained`) mimics off-the-shelf
counterfactual tools: every model input may change, cost is standardised L1 distance,
and there is no budget. It runs in two variants, with all features mutable and with
capital gain/loss held fixed, so that the violation rate is not driven by one feature.

## Limitations

- Adult is a proxy. Its label (income) is not an actual benefit or eligibility
  decision, and the data is from 1994.
- Action costs are assumptions. Changing jobs or adding schooling is harder for some
  people than others. A uniform cost function probably *understates* disparities.
- Sampling tops up small groups to `min_group_size`. This slightly over-represents
  those groups' members in the other attributes' slices.
- The recourse is for the model as trained. If the model is retrained, a
  recommended action may no longer lead to approval.

## License

MIT. See [LICENSE](LICENSE).
