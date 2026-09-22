"""
Deterministic chart generation (Phase 0).

Uses the non-interactive 'Agg' backend explicitly, since this runs inside a
server process (FastAPI / MCP server) with no display -- attempting to use
an interactive backend there raises or hangs.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from backend.tools.errors import PlottingError

_METRIC_Y_LABELS = {
    "normalized_returns": "Normalized Price (Start = 100)",
    "normalized_price": "Normalized Price (Start = 100)",
    "cumulative_returns": "Cumulative Return",
    "price": "Price (USD)",
}


def _ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_comparison(
    data: pd.DataFrame,
    *,
    metric: str = "normalized_returns",
    title: str | None = None,
    output_dir: str = "output",
    filename: str | None = None,
) -> str:
    """Plot one line per column of `data` (a DataFrame as produced by
    analysis.compute_comparison) and save it as a PNG. Returns the file path.
    """
    if data is None or data.empty:
        raise PlottingError("Cannot plot an empty dataset.", code="empty_data")

    out_dir = _ensure_output_dir(output_dir)
    filename = filename or f"comparison_{uuid.uuid4().hex[:10]}.png"
    out_path = out_dir / filename

    try:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        for column in data.columns:
            ax.plot(data.index, data[column], label=column, linewidth=1.8)

        y_label = _METRIC_Y_LABELS.get(metric, metric)
        ax.set_ylabel(y_label)
        ax.set_xlabel("Date")
        ax.set_title(title or f"{y_label} — {', '.join(data.columns)}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_path)
        return str(out_path)
    except Exception as exc:
        raise PlottingError(f"Failed to render comparison chart: {exc}") from exc
    finally:
        plt.close("all")


def plot_single(
    series: pd.Series,
    *,
    ticker: str,
    metric: str = "price",
    output_dir: str = "output",
    filename: str | None = None,
) -> str:
    """Plot a single ticker's series and save it as a PNG. Returns the file path."""
    if series is None or series.empty:
        raise PlottingError("Cannot plot an empty series.", code="empty_data")

    out_dir = _ensure_output_dir(output_dir)
    filename = filename or f"{ticker.lower()}_{uuid.uuid4().hex[:10]}.png"
    out_path = out_dir / filename

    try:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        ax.plot(series.index, series.values, linewidth=1.8, color="#1f77b4")
        y_label = _METRIC_Y_LABELS.get(metric, metric)
        ax.set_ylabel(y_label)
        ax.set_xlabel("Date")
        ax.set_title(f"{ticker} — {y_label}")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_path)
        return str(out_path)
    except Exception as exc:
        raise PlottingError(f"Failed to render chart for {ticker}: {exc}") from exc
    finally:
        plt.close("all")
