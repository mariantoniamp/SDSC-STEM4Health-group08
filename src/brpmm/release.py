"""Batch-release assistant: the leaderboard models turned into a decision-support report.

One entry point that brings the team's pieces together (no new model):
  Process A  the submitted ensemble (brpmm.predict): label, score, traffic light, top reasons
             + single-sample check (J&J: "one bad sample rejects the batch")
             + early warning (the same ensemble on the first third of each batch)
             + a cost table for the cut-off, so QA can choose the operating point
  Process B  aeration rule + accepted-only monitor with review flags (brpmm.process_b)
Every batch gets one action: release to QC / expert review / investigate. The action never
changes the leaderboard label; it tells people where to look.

Inputs:  calibration and test data of both processes (data_dir()).
Outputs: outputs/<time>_release_report.csv      one row per test batch, with its action
         outputs/<time>_operating_points.csv    cut-off shift -> expected errors (process A)
         outputs/<time>_release_summary.md      counts per action, for the reviewer

Run: python -m brpmm.release [--cutoff-shift 0.0]
"""

from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from brpmm.config import (
    BATCH_COL,
    CV_FOLDS,
    CV_REPEATS,
    LABEL_ACCEPTED,
    LABEL_REJECTED,
    OUTPUTS_DIR,
    SEED,
)
from brpmm.data import load_calibration, load_test, sensor_columns
from brpmm.evaluate import choose_cutoff, load_features
from brpmm.features import batch_features
from brpmm.methods import METHODS
from brpmm.metrics import leaderboard_f1
from brpmm.predict import cv_cutoff, predict_process
from brpmm.process_b import ProcessBModel

PROGRESS_BINS = 50  # 2 % of batch progress: "normal for this moment of the batch"
SHORT_BATCH = 2550  # process A has two batch types (~2,300-2,500 vs ~2,700 steps); separate limits
EARLY_FRACTION = 1 / 3  # early warning on the first third: 26 / 40 bad batches flagged in CV
FLAG_QUANTILE = 0.90  # sample check flags a batch above the 90th percentile of accepted batches
SHIFTS = (-0.2, -0.1, 0.0, 0.1, 0.2)  # cut-off shifts shown to QA in the cost table
ACTIONS = {"green": "release to QC", "amber": "expert review", "red": "investigate"}


# ---- 1. cost table: what each cut-off costs (process A, cross-validation) ----------------------
def operating_points(X: pd.DataFrame, y: pd.Series, shifts=SHIFTS) -> pd.DataFrame:
    """Out-of-fold ensemble scores on the CV splits; for each shift of the cut-off, the expected
    number of good batches thrown away and bad batches let through per 161 batches. F1 counts
    both errors equally; a plant may not, so QA picks the row from its own costs."""
    fit, score, _ = METHODS["ensemble"]
    rows = []
    for r in range(CV_REPEATS):
        oof = np.empty(len(X))
        for tr, te in StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED + r).split(X, y):
            oof[te] = score(fit(X.iloc[tr], y.iloc[tr]), X.iloc[te])
        base = choose_cutoff(
            oof, y
        )  # F1-optimal on these scores (optimistic, but same for all rows)
        rej = (y == LABEL_REJECTED).to_numpy()
        for s in shifts:
            pred = oof > base + s
            rows.append(
                dict(
                    shift=s,
                    good_thrown_away=(pred & ~rej).sum(),
                    bad_let_through=(~pred & rej).sum(),
                    f1=leaderboard_f1(y, np.where(pred, LABEL_REJECTED, LABEL_ACCEPTED)),
                )
            )
    t = pd.DataFrame(rows).groupby("shift").mean()
    return (t * [161 / len(X), 161 / len(X), 1]).round(3).reset_index()


# ---- 2. single-sample check: J&J's rule, one bad sample rejects the batch ---------------------
def _keys(ts: pd.DataFrame, by_length: bool) -> np.ndarray:
    pos = ts.groupby(BATCH_COL, sort=False).cumcount().to_numpy()
    n = ts.groupby(BATCH_COL, sort=False)[BATCH_COL].transform("size").to_numpy()
    kind = (n < SHORT_BATCH).astype(int) if by_length else np.zeros(len(ts), int)
    return kind * PROGRESS_BINS + np.minimum(pos * PROGRESS_BINS // n, PROGRESS_BINS - 1)


def _limits(ts: pd.DataFrame, keys: np.ndarray, batches) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = ts[BATCH_COL].isin(batches).to_numpy()
    g = ts.loc[m, sensor_columns(ts)].assign(_k=keys[m]).groupby("_k")
    return g.min(), g.max()


def _outside(
    ts: pd.DataFrame, keys: np.ndarray, lo: pd.DataFrame, hi: pd.DataFrame
) -> pd.DataFrame:
    """Per batch: samples outside the accepted range for their moment of the batch, the sensor
    with most of them, and the batch progress (%) where they concentrate."""
    S = sensor_columns(ts)
    ok = np.isin(keys, lo.index)
    X = ts[S].to_numpy(float)[ok]
    k = keys[ok]
    out = (lo.loc[k].to_numpy() > X) | (hi.loc[k].to_numpy() < X)
    d = pd.DataFrame(out, columns=S).assign(
        **{BATCH_COL: ts[BATCH_COL].to_numpy()[ok], "_p": (k % PROGRESS_BINS) * 2}
    )
    any_out = d[S].any(axis=1)
    res = pd.DataFrame({"samples_outside": any_out.groupby(d[BATCH_COL], sort=False).sum()})
    res["worst_sensor"] = d.groupby(BATCH_COL, sort=False)[S].sum().idxmax(axis=1)
    res["at_progress_pct"] = d[any_out].groupby(BATCH_COL, sort=False)["_p"].median()
    res.loc[res.samples_outside == 0, ["worst_sensor", "at_progress_pct"]] = np.nan
    return res


def sample_check(
    ts_cal: pd.DataFrame, labels: pd.Series, ts_new: pd.DataFrame, by_length: bool
) -> pd.DataFrame:
    """Limits from accepted calibration batches only. The flag threshold is the 90th percentile of
    accepted batches, each scored against limits built without its own group (5 groups), so it
    is set the way a new batch is scored. Calibration AUC of samples_outside: 0.83 (process A)."""
    acc = labels.index[labels == LABEL_ACCEPTED].to_numpy()
    kc = _keys(ts_cal, by_length)
    held = []
    for ref, tgt in KFold(5, shuffle=True, random_state=SEED).split(acc):
        lo, hi = _limits(ts_cal, kc, acc[ref])
        part = ts_cal[ts_cal[BATCH_COL].isin(acc[tgt])]
        held.append(_outside(part, _keys(part, by_length), lo, hi))
    threshold = pd.concat(held).samples_outside.quantile(FLAG_QUANTILE)
    lo, hi = _limits(ts_cal, kc, acc)
    res = _outside(ts_new, _keys(ts_new, by_length), lo, hi)
    res["sample_flag"] = res.samples_outside > threshold
    return res


# ---- 3. early warning: the same ensemble on the first third of each batch ---------------------
def _first_part(ts: pd.DataFrame, frac: float) -> pd.DataFrame:
    pos = ts.groupby(BATCH_COL, sort=False).cumcount()
    n = ts.groupby(BATCH_COL, sort=False)[BATCH_COL].transform("size")
    return ts[pos < frac * n]


def early_warning(ts_cal: pd.DataFrame, labels: pd.Series, ts_new: pd.DataFrame) -> pd.Series:
    fit, score, _ = METHODS["ensemble"]
    X = batch_features(_first_part(ts_cal, EARLY_FRACTION))
    y = labels.reindex(X.index)
    Xn = batch_features(_first_part(ts_new, EARLY_FRACTION)).reindex(columns=X.columns)
    cut = cv_cutoff(fit, score, X, y)
    return pd.Series(score(fit(X, y), Xn) > cut, index=Xn.index, name="early_flag")


# ---- the report ------------------------------------------------------------------------------
def report_a(cutoff_shift: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts_cal, labels = load_calibration("A")
    ts_new = load_test("A")
    rep = predict_process(
        "A"
    )  # label, score, cutoff, light, top_reasons: same code as the submission
    if cutoff_shift:
        rep["cutoff"] = rep["cutoff"] + cutoff_shift
        rep["label"] = np.where(rep["score"] > rep["cutoff"], LABEL_REJECTED, LABEL_ACCEPTED)
    rep = rep.join(sample_check(ts_cal, labels, ts_new, by_length=True)).join(
        early_warning(ts_cal, labels, ts_new)
    )
    action = rep["light"].map(ACTIONS)
    escalate = (rep["light"] == "green") & rep["sample_flag"]  # model says fine, samples say look
    action[escalate] = "expert review"
    rep["action"] = action
    rep["why"] = np.where(
        escalate,
        "samples outside the normal range for their point in the batch",
        rep["top_reasons"],
    )
    X, y = load_features("A")
    return rep, operating_points(X, y)


def report_b() -> pd.DataFrame:
    ts_cal, labels = load_calibration("B")
    ts_new = load_test("B")
    out = ProcessBModel().fit(ts_cal, labels).predict(ts_new)
    out["project"] = "B"
    out["action"] = np.select(
        [out["label"] == LABEL_REJECTED, out["review"]],
        ["investigate", "expert review"],
        "release to QC",
    )
    out["why"] = out["reason"]
    return out[["project", "label", "action", "why", "start_outside_calibration"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--cutoff-shift",
        type=float,
        default=0.0,
        help="process A: move the cut-off (positive = reject less)",
    )
    args = ap.parse_args()
    a, ops = report_a(args.cutoff_shift)
    rep = pd.concat([a, report_b()], sort=False)
    rep.index.name = BATCH_COL
    keep = [
        "project",
        "label",
        "action",
        "why",
        "light",
        "score",
        "cutoff",
        "samples_outside",
        "worst_sensor",
        "at_progress_pct",
        "sample_flag",
        "early_flag",
        "top_reasons",
        "start_outside_calibration",
    ]
    rep = rep[[c for c in keep if c in rep.columns]]

    # ---- PLUMBING: write the three files ----
    OUTPUTS_DIR.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    rep.reset_index().to_csv(OUTPUTS_DIR / f"{stamp}_release_report.csv", index=False)
    ops.to_csv(OUTPUTS_DIR / f"{stamp}_operating_points.csv", index=False)
    counts = pd.crosstab(rep["project"], rep["action"])
    md = [
        f"# Batch-release report ({stamp})",
        "",
        f"Process A cut-off shift: {args.cutoff_shift:+.2f}",
        "",
        "## Actions per process",
        "",
        "```",
        counts.to_string(),
        "```",
        "",
        "## Process A: cost of each cut-off (cross-validation, per 161 batches)",
        "",
        "```",
        ops.to_string(index=False),
        "```",
        "",
        f"Early warning (first third of the batch) flags {int(a['early_flag'].sum())} "
        f"of {len(a)} process A test batches.",
    ]
    (OUTPUTS_DIR / f"{stamp}_release_summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
