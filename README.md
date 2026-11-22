# SDSC-STEM4Health-group08 — Building Resilient Process Monitoring Models (Project 2)

Batch process monitoring for two processes, built for the STEM4Health x Johnson & Johnson hackathon:
**A** is an industrial process with imbalanced labels, and **B** is a simulated fed-batch fermentation with strong dynamics.
We use one monitoring strategy for both processes and fit it separately to each.

## Quick start

```bash
make setup          # Renku: editable install + git hooks (re-run every new session); use `make setup-local` elsewhere
make data           # stage challenge data into data/raw (once per machine)
cp .env.example .env  # add leaderboard team name / token
make check          # ruff + pytest
make predict        # LEADERBOARD FILE (python -m brpmm.predict) -> outputs/<time>_A-ensemble_B-pca_NOT-SUBMITTED.csv
make release        # INDUSTRY USE (python -m brpmm.release) -> one action per test batch + cut-off cost table
make monitor        # accepted-only solution (round 2) -> outputs/<time>_monitor.csv + .json
make evaluate       # every method on the shared CV protocol -> findings report
make select         # evidence for the A model (~2 min) -> outputs/process_A/final_selection.json
make tune           # nested tuning + extra views (~25 min) -> outputs/process_A/tuning/
make eda-a          # process A data analysis -> outputs/process_A/eda/REPORT.md
```

Nothing is ever submitted automatically; `brpmm.submission.submit` is a separate, deliberate step.

```python
from brpmm.data import load_calibration, load_test

ts, labels = load_calibration("B")  # long format: batch_id, t, sensors...
```

## Results

| | Model | Calibration CV (F1, accepted positive) | Leaderboard |
|---|---|---|---|
| A | `methods.ensemble`: RBF SVM + kNN on PCA + one-class SVM on 22 median/std features | 0.941 ± 0.002 (fresh splits) | **0.931** |
| B | `methods.pca_mspc` on accepted batches (same 5 rejects as the aeration rule) | 1.000 | **1.000** |
| A, accepted only | `scripts/monitor.py`: PCA + kNN monitors | ROC-AUC 0.842; catches 46 % at 94 % specificity | not scored |
| B, accepted only | `process_b`: aeration rule + PCA monitor with review flags | 1.000, 5/5, 0 false alarms | not scored |

Every model, its validation type and its score: `docs/model_results.md`.

## Layout

```
src/brpmm/        reusable code
  config.py         paths (data/raw first, then Renku input-data, then challenge_docs) + model/CV settings
  data.py           load calibration/test time series + labels
  submission.py     build / validate / submit leaderboard files
  -- leaderboard pipeline --
  features.py       batch features: level, shape, within-batch correlations (243 A / 393 B)
  methods.py        every compared method as fit / score / explain; final A `ensemble`
  evaluate.py       the shared CV protocol (seeds 42+r, 5x5 folds, inner 3-fold cut-off)
  metrics.py        leaderboard F1 on label strings (accepted = positive)
  predict.py        final models on all calibration data -> submission + per-batch details
  release.py        batch-release assistant: action per batch, flags, reasons, early warning
  report.py         findings report
  -- accepted-only monitoring, analysis and evidence --
  prepare.py        stage raw data into data/raw
  preprocessing.py  segmentation (start/steady/end), gross-outlier mask, quality report
  steady_features.py steady-state batch features + model-space transforms
  mspc.py, novelty.py  accepted-only monitors: PCA (T², SPE, contributions), kNN distance
  rules.py, process_b.py  process B: aeration rule + accepted-only monitor with review flags
  models.py         alternative A ensembles compared against the submitted one (SETTINGS)
  evaluation.py     CV tools: nested thresholds, bootstrap, permutation, importance
docs/model_results.md  team registry of every model and its validated score
scripts/
  monitor.py          accepted-only monitoring for A and B (round 2)
  select_final_a.py   submitted A model vs its alternatives, fresh splits
  tune_process_a.py   nested hyperparameter search; extra views under the same rule
  eda_process_a.py    process A data analysis report + figures
  dashboard_process_a.py  datasets for the interactive analysis page
notebooks/        exploration — prefix with initials + number, e.g. 01_mv_eda_process_b.ipynb
tests/            pytest, synthetic data only
data/             gitignored: raw/ (staged challenge data), processed/ (feature tables)
outputs/          gitignored; reports, figures, timestamped submissions
challenge_docs/   gitignored; local copy of the brief + data (not shared via git)
```

## Batch-release assistant (industry use)

`python -m brpmm.release` (or `make release`, ~30 s) turns the leaderboard models into a decision-support
report. It adds no new model: it reuses `predict.py` (process A ensemble) and `process_b.py` (process B rule
and monitor). Each test batch gets **one action: release to QC, expert review or investigate**. The action
never changes the leaderboard label; it tells people where to look.

| Proposal | What it does | Evidence (cross-validation, process A) |
|---|---|---|
| Traffic light | green = the 3 ensemble members accept, red = all reject, amber = they disagree → expert review | amber = 22 % of batches, 45 % of all errors |
| Reasons | top 3 features furthest from accepted batches, per batch | temperature_1, density, temperature_4 decide most cases |
| Single-sample check | J&J: "one bad sample rejects the batch". Each sample vs the accepted range **for its moment of the batch** (per batch type, 2 % of progress); a green batch above the accepted 90th percentile is escalated to review | separates good from bad with AUC 0.83 |
| Early warning | the same ensemble on the first third of each batch | 26 / 40 bad batches flagged, no extra false alarms |
| Cut-off cost table | expected good batches thrown away vs bad let through for cut-off shifts; QA picks the row (`--cutoff-shift`) | F1 counts both errors equally, a plant may not |
| Process B review flags | rule rejects the aeration fault; batches started outside the calibrated range are flagged for review, not rejected | 5 / 5 faults, 0 false alarms |

Outputs (in `outputs/`, timestamped): `_release_report.csv` (one row per test batch), `_operating_points.csv`,
`_release_summary.md`. The report also states where the score stops: about 7 rejected calibration batches
look normal on all 11 sensors (see `docs/model_results.md`), so the next gain is data (lab results, recipe),
not models.

## Conventions

- Code that more than one notebook needs goes in `src/brpmm`. Notebooks import it and don't copy it.
- Notebook outputs are stripped on commit (nbstripout) and files over 1 MB are blocked (pre-commit).
- Every leaderboard submission is saved via `brpmm.submission.save_submission`. Write down what produced it.

## Scoring

Score = mean of the per-process F1 (A and B weighted equally). The brief says F1 is for the
rejected class, but every leaderboard entry's accuracy, precision and recall solve to **120
batches in the scored class** of the 161-batch A test set (calibration: 121 accepted, 40 rejected;
test features match calibration). So the F1 on the leaderboard behaves as **F1 of the accepted
class**: a missed reject and a false alarm cost about the same, and predicting "all accepted"
already scores 0.854 on A. Our own submission confirmed it: its returned accuracy, precision and
recall fit only this reading (30 of 41 rejects caught, 6 of 120 good batches thrown away).
`brpmm.metrics.leaderboard_f1` and `brpmm.evaluation` (`f1_accepted`) compute it identically
(tests/test_metric_alignment.py).

---

## Getting started in the SDSC STEM4Health Hackathon

Welcome to the SDSC STEM4Health Hackathon! This git repository is your team's shared codebase: use it to keep your code in sync and under version control.

It contains an `environment.yml` file listing the packages your session starts with, including pandas, numpy, scikit-learn, plotly, transformers and LangChain. Apart from that, the repository is yours: organise it however suits your team.

&nbsp;

## 1. Start working

### 1.1 Launch a session
On your project page, in the **Launchers** section, click the green **Launch** button. Choose **JupyterLab** or **VSCodium**, whichever you prefer.

&nbsp;

### 1.2 Access the input data
Your session has an `input-data` folder with one subfolder per challenge. Use the data for the challenge your team is working on.

> **Note:** Don't commit the input data to the repository. Read it straight from `input-data`.

&nbsp;

### 1.3 Connect to GitHub
To push your work, connect your Renku account to your GitHub account (see the [Renku documentation](https://docs.renkulab.io/)). You already have access to the repository we created for your team. It is cloned into your session automatically.

&nbsp;

### 1.4 Organise your repository
You choose the structure. For example:

```bash
.
├── environment.yml
├── notebooks/
├── src/
├── outputs/
└── README.md
```

Agree on the layout with your team early on. That way everyone knows where to put their files.

&nbsp;

### 1.5 Work with your team using git
Everyone works in their own session and shares work through the repository:

```bash
git pull                        # get your teammates' latest changes
git add <files>
git commit -m "Short description of your change"
git push                        # share your changes
```

Tips:
- Pull before you start working, and push often.
- To avoid conflicts, try not to edit the same notebook as a teammate at the same time.
- Keep large files (data, model weights) out of git in the team-workspace folder.

> **Important:** Anything you don't commit and push is lost when the session shuts down.

&nbsp;

### 1.6 Add or change packages
**Quick test (temporary):** install a package from a terminal or notebook cell:

```bash
pip install <package-name>
```

These packages are lost when the session shuts down.

**Permanent (for the whole team):** add the package to `environment.yml`, then commit and push. Rebuild the session environment from your launcher, and every **new** session will include it.

```yaml
dependencies:
  - pandas
  - <new-conda-package>
  - pip:
      - <new-pip-package>
```

&nbsp;

### 1.7 Use AI coding assistants (optional)
Your session comes with several AI coding assistants. They all run in the terminal:

| Assistant | Command |
|---|---|
| GitHub Copilot CLI | `copilot` |
| Claude Code | `claude` |
| OpenAI Codex CLI | `codex` |
| Pi | `pi` |

1. Open a **Terminal** in your session.
2. Run the command for the assistant you want to use, for example:
   ```bash
   copilot
   ```
3. When prompted, sign in with your own account or API key for that tool. Access and any usage costs go through your own account.

> **Note:** These sign-ins are separate from your Renku login. Sign-ins may not survive a session shutdown, so you might need to sign in again.

&nbsp;

### 1.8 Change your compute resources
If you need more CPU, or more memory or a GPU:

1. **Pause** your session (top-left corner). Your files and changes are kept.
2. From the dropdown on the launcher button, choose **Modify session resources**.
3. Select a larger resource class and confirm.
4. **Resume** the session.

To change the resources for all future sessions, edit the **Default Resource Class** in your launcher (click the pencil icon). To change them for a single new session, use the dropdown next to **Launch** and choose **Custom launch**.

See the [Renku documentation](https://docs.renkulab.io/en/latest/docs/users/compute/guides/select-compute-resources-for-session/) for details.

&nbsp;

## 2. Pause or shut down your session

Sessions keep running when you close your browser. Please free up resources when you finish:

* **Pause session** (top-left corner): your workspace is kept for two weeks. Sessions also pause automatically after 30 minutes of inactivity.

![Pause button](https://raw.githubusercontent.com/SwissDataScienceCenter/renku-outreach/refs/heads/main/tutorial/pause_button.jpeg)

&nbsp;

* **Shut down session** (top-left corner): only work you have **committed and pushed** to the repository is kept. Everything else is lost.

![Shut down button](https://raw.githubusercontent.com/SwissDataScienceCenter/renku-outreach/refs/heads/main/tutorial/shutdown_button.jpeg)

> **Important:** Even paused sessions can end unexpectedly. **Commit and push regularly.**

&nbsp;

## Need help?
Ask the SDSC coordinators or the Renku team on site, or write to `hello@renku.io` with the subject *[SDSC STEM4Health] Your topic*.
