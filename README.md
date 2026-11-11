# Peace as a Coupled Feedback System — replication package

Version 2.5, November 11, 2026. Companion to Chapter 11 and Appendix D of S. Massoud Amin, *The Coupled World* (manuscript, 2026), and to the paper "The Feedback Dynamics of Peace: Loops, Thresholds, and Conflict Onset" (manuscript, November 2026).

Everything here is public. The derived panels are shipped, so the yearly results can be checked without downloading any source data. The remaining results need the public source files listed in `data/README.md`.

## Layout

```
code/      Python scripts (run from the repository root)
panels/    derived country-year, pair-year, and survey panels
outputs/   recorded output of the runs reported in the book
figures/   figures of the paper and the book
data/      empty; put the downloaded source files here (see data/README.md)
```

## Start here: results that run from the shipped panels

```
pip install -r requirements.txt
python3 code/table_11_2_from_panels.py
python3 code/trust_pair_from_panels.py
python3 code/trust_forecast_check.py
python3 code/peace_game.py
python3 code/state_estimators.py
python3 code/estimator_ranking.py
python3 code/monthly_panel.py      # needs the source files listed in data/README.md
python3 code/monthly_observer.py   # about twenty minutes
python3 code/fig4.py
```

| Script | Reads | Reproduces in the book |
|---|---|---|
| `table_11_2_from_panels.py` | `panels/panel2.csv` | Table 11.2; Table D.2; the held-out checks of Chapter 11 and Appendix D (base rate, Brier score, calibration by thirds, top-tenth capture, bootstrap by country, permutation test, escalation baseline) |
| `trust_pair_from_panels.py` | `panels/panel2.csv`, `wvs7_country_measures_weighted.csv`, `pair_year_panel.csv` | Table 11.3; Table 11.4; the trust rank correlation of Figure 11.3; the pair escalation ladder, pair escalation AUC, and reciprocity coefficients |
| `trust_forecast_check.py` | `panels/panel2.csv`, `wvs7_country_measures_weighted.csv` | the held-out check of Chapter 11 on whether trust and its product with rule of law improve the yearly onset forecast |
| `peace_game.py` | none | the laboratory game of Chapter 11 and Appendix D (Table D.6), run with simulated teams; illustrative coefficients, no data files |
| `state_estimators.py` | `panels/panel2.csv` | sixteen state estimators for the violence state (Kalman, extended, unscented and Tobit Kalman filters, particle filter, hidden Markov model, Markov chain, hurdle model, gradient boosting, random forest, feed-forward, radial-basis-function, wavelet + RBF, recurrent high-order, echo state and gated recurrent networks), their ensemble, three logistic baselines, and a known-truth test, for one test window (the gated recurrent network needs PyTorch and is skipped without it) |
| `estimator_ranking.py` | `panels/panel2.csv` | runs `state_estimators.py` on four rolling test windows and ranks the estimators (Table D.8, Figure D.2) |
| `monthly_panel.py` | UCDP GED 26.1 and Candidate Events (January-September 2026), ACLED weekly aggregates, World Bank Real Time Food Prices, `panels/panel2.csv` | builds the country-month panel `panels/monthly_panel.csv`, January 1989 to September 2026 (the panel is not shipped: it contains counts derived from ACLED, whose terms govern redistribution) |
| `monthly_observer.py` | `panels/monthly_panel.csv` | a monthly Tobit Kalman filter with one, three and six sensors, three logistic regressions and gradient boosting, on four rolling test windows and horizons of 3, 6 and 12 months (Table D.9; Chapter 15, open problem 11) |
| `fig4.py` | `panels/testwars_rank.csv`, `rank2026.csv` | the test-war ranks and the 2026 ranking |
| `fig2.py`, `fig_loops.py`, `peace_eq2.py` | nothing | the control-loop, causal-loop, and equation-sheet figures (the equation sheet carries the paper's numbering; the book numbers the same conditions (11.1) to (11.5)) |

Their recorded output is in `outputs/`.

## Results that need the source files

| Script | Source files (see `data/README.md`) | Writes | Reported in the book |
|---|---|---|---|
| `fit1.py` | UCDP country-year, FAO index, SIPRI | `panels/panel.csv` | onset and escalation models with food and military terms |
| `fit3.py` | `panels/panel.csv`, V-Dem, UCDP/PRIO | `panels/panel2.csv` | Table 11.2 (the original fit) |
| `fit2.py` | UCDP GED, FAO index | prints | Table 11.1, country-month columns; the count of eleven silent onsets in nineteen |
| `pair.py` | UCDP GED, UCDP Dyadic | `panels/pair_year_panel.csv` | Table 11.1, pair-month columns; pair escalation; reciprocity; the interstate subset |
| `trust.py` | WVS wave 7, UCDP GED, `panels/panel2.csv` | `panels/wvs7_country_measures_weighted.csv`, `wvs7_regions.csv` | Tables 11.3 and 11.4; Figures 11.3 and 11.4; the regional test |
| `loops4.py` | World Bank indicators, UCDP GED, `adjacency.csv`, `panels/panel2.csv` | `panels/panel_loops.csv` | youth share, contagion, population growth, drought |
| `food.py`, `food2.py`, `food3.py` | WFP prices, World Bank inflation, UCDP GED | `panels/staple_chg.csv`, `panel3.csv` | food-price tests |
| `res.py` | World Bank reserves, `panels/panel3.csv` | `panels/panel4.csv` | reserves and the buffer interaction |
| `tone.py`, `tone2.py` | ICEWS, UCDP GED, `panels/panel4.csv` | `panels/tone_all.csv`, `panel5.csv` | press-hostility tests |
| `predict.py` | `panels/panel2.csv`, UCDP, V-Dem | `panels/testwars_rank.csv`, `rank2026.csv` | test-war ranks and the 2026 ranking |
| `silence_sensitivity.py` | UCDP GED | prints | Table D.1 |
| `ucdp_family_tests.py` | UCDP Peace Agreement, External Support, MIC, DECO, Battle-Related Deaths, Dyadic; `panels/panel2.csv` | `outputs/out_*.csv` | Table D.4 |
| `ucdp_family_tests2.py` | UCDP Conflict Termination, Onset, Conflict Issues, Battle-Related Deaths, Dyadic; `panels/panel2.csv` | `outputs/out_*.csv` | Table D.4; the check of the panel's onset coding |
| `acled_second_record.py` | UCDP GED, ACLED weekly aggregates | prints; one csv | Table D.5 |

Run order for a full rebuild of the panels: `fit1`, `fit3`, `food`, `food2`, `food3`, `res`, `tone`, `tone2`, `fit2`, `predict`, `pair`, `trust`, `loops4`. The last four scripts in the table take optional arguments and default to the files in `data/`: `silence_sensitivity.py [GED csv]`, `ucdp_family_tests.py [data directory] [repository root]`, `ucdp_family_tests2.py [data directory] [repository root]`, `acled_second_record.py [GED csv] [ACLED folder]`.

No fitted model objects are stored. Each script refits when it is run.

## What has been checked, and what has not

On November 10, 2026, in a fresh environment, every script was compiled, and the scripts that run from the shipped panels were run. They reproduce Table 11.2 exactly, Table D.2 to within 0.01, and the counts of Tables 11.3 and 11.4. The scripts that need source files were run on those files when the results were first produced (September and October 2026), and their recorded output is in `outputs/`. For this release the input file names in those scripts were made uniform and matched to `data/README.md`; they have not been rerun against the source files since that change. A rebuild of the panels from the source files by a second party is the independent test that remains.

## Definitions

Yearly: each row is a country-year. Predictors use that year or earlier, and the outcome is the following year. Models are fitted on rows 1991–2020 and tested on rows 2021–2024, whose outcomes are the onsets of 2022–2025. A war onset is a country-year with at least 1,000 state-based battle deaths after two years below that level; a conflict onset uses 25 deaths.

Monthly (`fit2.py`): the first month with at least 25 state-based battle deaths after twenty-four months each below 25, followed by at least 1,000 deaths within twelve months. Only UCDP GED state-based events (`type_of_violence == 1`) are counted, and a month with no event is entered as zero. `silence_sensitivity.py` recomputes the count of onsets preceded by six months with no recorded deaths under other definitions.

## Forecast on record

`panels/rank2026.csv` is the model's ranking for 2026, made from 2025 data on September 24, 2026, before any 2026 figures were available. Score it against UCDP's 2026 release.
