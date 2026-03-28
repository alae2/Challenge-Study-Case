"""
utils.py — Shared utilities for Wind Power Forecasting
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Shared constants (must match preprocessing notebook) ──────────────────────

DATA_DIR   = Path("data")
TEST_START = pd.Timestamp('2025-11-01 00:00', tz='UTC')
N_SPLITS   = 6
VAL_DAYS   = 90


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir=DATA_DIR):
    """
    Load all parquet files and site mapping.

    Returns
    -------
    X_train, y_train, X_test, y_test : pd.DataFrame
    meta_train, meta_test            : pd.DataFrame  (delivery_time, site_name, site_id, installed_capacity)
    site_mapping                     : dict  {site_id (int): site_name (str)}
    """

    X_train = pd.read_parquet(data_dir / "X_train.parquet")
    y_train = pd.read_parquet(data_dir / "y_train.parquet").squeeze()
    X_test  = pd.read_parquet(data_dir / "X_test.parquet")
    y_test  = pd.read_parquet(data_dir / "y_test.parquet").squeeze()

    meta_train = pd.read_parquet(data_dir / "meta_train.parquet")
    meta_test  = pd.read_parquet(data_dir / "meta_test.parquet")

    with open(data_dir / "site_mapping.json") as f:
        site_mapping = {int(k): v for k, v in json.load(f).items()}

    print(f"X_train : {X_train.shape}  |  X_test : {X_test.shape}")
    print(f"Train period : {meta_train['delivery_time'].min().date()} "
          f"→ {meta_train['delivery_time'].max().date()}")
    print(f"Test  period : {meta_test['delivery_time'].min().date()} "
          f"→ {meta_test['delivery_time'].max().date()}")
    print(f"Sites : {len(site_mapping)}")

    return X_train, y_train, X_test, y_test, meta_train, meta_test, site_mapping


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cross-validation folds
# ─────────────────────────────────────────────────────────────────────────────

def get_cv_folds(delivery_times, n_splits=N_SPLITS, val_days=VAL_DAYS):
    """
    Reproduce the CV strategy :
      - Expanding window (train always starts from the beginning)
      - Validation windows slide backward from the end of the training set
      - Each val window = val_days days
      - Folds returned in chronological order (oldest val first → fold 1)

    Parameters
    ----------
    delivery_times : pd.Series
        Timestamps of the training set rows (same index as X_train / y_train).
        Must be sorted chronologically and tz-aware.
    n_splits : int
        Number of CV folds (default 6).
    val_days : int
        Validation window size in days (default 90 ≈ 3 months).

    Returns
    -------
    folds : list of (tr_idx, val_idx)
        Integer index arrays into the training set, one tuple per fold.
        Ordered chronologically: fold 0 has the oldest val window.
    """
    delivery_times = pd.Series(delivery_times).reset_index(drop=True)
    val_duration   = pd.Timedelta(days=val_days)
    max_time       = delivery_times.max()

    raw_folds = []
    for fold in range(n_splits):
        val_end   = max_time - fold * val_duration
        val_start = val_end  - val_duration

        train_mask = delivery_times < val_start
        val_mask   = (delivery_times >= val_start) & (delivery_times < val_end)

        tr_idx  = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        if len(tr_idx) == 0 or len(val_idx) == 0:
            continue

        raw_folds.append((tr_idx, val_idx, val_start, val_end))

    # Reverse so fold 0 = oldest val window (chronological order)
    folds = [(tr, val) for tr, val, _, _ in reversed(raw_folds)]

    print(f"CV: {len(folds)} folds | val window = {val_days} days | expanding train")
    for i, (tr_idx, val_idx) in enumerate(folds):
        t0 = delivery_times.iloc[tr_idx[0]].date()
        t1 = delivery_times.iloc[tr_idx[-1]].date()
        v0 = delivery_times.iloc[val_idx[0]].date()
        v1 = delivery_times.iloc[val_idx[-1]].date()
        print(f"  Fold {i+1}: train {t0} → {t1}  |  val {v0} → {v1}"
                f"  ({len(tr_idx):,} / {len(val_idx):,} rows)")

    return folds


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred):
    """Compute MAE, RMSE, nRMSE (normalised by range of y_true)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = y_true.max() - y_true.min()
    nrmse = rmse / denom if denom > 1e-9 else np.nan
    return {"MAE": mae, "RMSE": rmse, "nRMSE": nrmse, "n": len(y_true)}


def evaluate(y_true, y_pred, site_names=None, model_name="Model"):
    """
    Compute MAE, RMSE, nRMSE — globally and per site.

    Parameters
    ----------
    y_true      : array-like  (capacity_factor ground truth)
    y_pred      : array-like  (capacity_factor predictions)
    site_names  : array-like or None
        Site name for each row. If None, only global metrics are returned.
    model_name  : str  (for display)

    Returns
    -------
    global_metrics : dict   {MAE, RMSE, nRMSE, n}
    per_site       : pd.DataFrame or None
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    global_metrics = _metrics(y_true, y_pred)

    print(f"[{model_name}]  Global — "
            f"MAE={global_metrics['MAE']:.4f}  "
            f"RMSE={global_metrics['RMSE']:.4f}  "
            f"nRMSE={global_metrics['nRMSE']:.4f}  "
            f"(n={global_metrics['n']:,})")

    per_site = None
    if site_names is not None:
        site_names = np.asarray(site_names)
        rows = []
        for site in sorted(np.unique(site_names)):
            mask = site_names == site
            m = _metrics(y_true[mask], y_pred[mask])
            m["site"] = site
            rows.append(m)
        per_site = (
            pd.DataFrame(rows)
              .set_index("site")[["MAE", "RMSE", "nRMSE", "n"]]
              .sort_values("RMSE", ascending=False)
        )
        print(per_site.round(4).to_string())

    return global_metrics, per_site


def compare_models(results: dict):
    """
    Compare global metrics across multiple models.

    Parameters
    ----------
    results : dict  {model_name: global_metrics_dict}
        Output of evaluate() for each model.

    Returns
    -------
    pd.DataFrame  sorted by RMSE ascending.
    """
    rows = [{"model": name, **metrics} for name, metrics in results.items()]
    df = pd.DataFrame(rows).set_index("model")[["MAE", "RMSE", "nRMSE"]]
    df = df.sort_values("RMSE")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3))
    for ax, metric in zip(axes, ["MAE", "RMSE", "nRMSE"]):
        vals = df[metric]
        colors = ["#2196F3" if i == 0 else "#90CAF9" for i in range(len(vals))]
        ax.barh(vals.index[::-1], vals.values[::-1], color=colors[::-1])
        ax.set_title(metric)
        ax.set_xlabel("Score (lower = better)")
    plt.suptitle("Model comparison", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()

    return df.round(4)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Forecast plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_forecast(delivery_times, y_true, y_pred, site_names,
                  model_name="Model", resample="D",
                  installed_capacity=None, ncols=2):
    """
    Plot actual vs predicted capacity factor over the test period, one panel per site.

    Parameters
    ----------
    delivery_times    : array-like of pd.Timestamp (tz-aware UTC)
    y_true            : array-like  (capacity_factor ground truth)
    y_pred            : array-like  (capacity_factor predictions)
    site_names        : array-like  (site name for each row)
    model_name        : str
    resample          : str  resampling frequency for readability
                        'H' = hourly (raw), 'D' = daily mean (default, less noisy)
    installed_capacity: dict or None  {site_name: MW}
                        If provided, adds a secondary y-axis in MW.
    ncols             : int  number of subplot columns (default 2)
    """
    delivery_times = pd.to_datetime(delivery_times, utc=True)
    y_true         = np.asarray(y_true,   dtype=float)
    y_pred         = np.asarray(y_pred,   dtype=float)
    site_names     = np.asarray(site_names)

    sites  = sorted(np.unique(site_names))
    nrows  = (len(sites) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), sharey=True)
    axes = np.array(axes).flatten()

    for i, site in enumerate(sites):
        ax   = axes[i]
        mask = site_names == site

        ts_true = pd.Series(y_true[mask], index=delivery_times[mask]).sort_index()
        ts_pred = pd.Series(y_pred[mask], index=delivery_times[mask]).sort_index()

        if resample != "H":
            ts_true = ts_true.resample(resample).mean()
            ts_pred = ts_pred.resample(resample).mean()

        ax.plot(ts_true.index, ts_true.values,
                lw=0.8, color="steelblue", label="Actual", alpha=0.9)
        ax.plot(ts_pred.index, ts_pred.values,
                lw=0.8, color="tomato",    label=model_name, alpha=0.85)

        # Per-site metrics in title
        m = _metrics(y_true[mask], y_pred[mask])
        ax.set_title(f"{site}\nMAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  nRMSE={m['nRMSE']:.3f}",
                     fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.set_ylabel("Capacity factor")

        if installed_capacity and site in installed_capacity:
            cap = installed_capacity[site]
            ax2 = ax.twinx()
            ax2.set_ylim(0, 1.05 * cap)
            ax2.set_ylabel("MW", fontsize=7, color="gray")
            ax2.tick_params(axis="y", labelsize=7, colors="gray")

        if i == 0:
            ax.legend(loc="upper left", fontsize=8)

    # Hide unused subplots
    for j in range(len(sites), len(axes)):
        axes[j].set_visible(False)

    freq_label = {"H": "hourly", "D": "daily mean", "W": "weekly mean"}.get(resample, resample)
    plt.suptitle(f"Forecast vs Actual — {model_name}  ({freq_label})", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_forecast_zoom(delivery_times, y_true, y_pred, site_names,
                       site, start, end, model_name="Model"):
    """
    Zoom into a specific site and time window — hourly resolution.

    Parameters
    ----------
    site  : str   site name
    start : str   e.g. '2025-11-01'
    end   : str   e.g. '2025-11-15'
    """
    delivery_times = pd.to_datetime(delivery_times, utc=True)
    site_names     = np.asarray(site_names)
    mask = (site_names == site)

    ts_true = pd.Series(np.asarray(y_true,  dtype=float)[mask],
                        index=delivery_times[mask]).sort_index()
    ts_pred = pd.Series(np.asarray(y_pred, dtype=float)[mask],
                        index=delivery_times[mask]).sort_index()

    ts_true = ts_true.loc[start:end]
    ts_pred = ts_pred.loc[start:end]

    m = _metrics(ts_true.values, ts_pred.values)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ts_true.index, ts_true.values, lw=1.2, color="steelblue", label="Actual")
    ax.plot(ts_pred.index, ts_pred.values, lw=1.2, color="tomato",    label=model_name, alpha=0.85)
    ax.fill_between(ts_true.index,
                    ts_true.values, ts_pred.values,
                    alpha=0.15, color="tomato")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Capacity factor")
    ax.set_title(f"{site} — {start} → {end}\n"
                 f"MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  nRMSE={m['nRMSE']:.3f}")
    ax.legend()
    plt.tight_layout()
    plt.show()