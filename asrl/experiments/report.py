"""Aggregate per-run results into summary tables and plots."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..benchmarks.variants import ABLATIONS, BENCHMARKS
from ..utils import read_jsonl

COLUMNS = [
    ("test_sharpe", "Sharpe", "{:+.3f}"),
    ("test_cumulative_return", "Cum. return", "{:+.1%}"),
    ("test_annual_volatility", "Ann. vol", "{:.1%}"),
    ("test_max_drawdown", "Max DD", "{:.1%}"),
    ("test_calibration_error_docx", "E_cal (DOCX)", "{:.3f}"),
    ("test_calibration_error_coverage", "E_cal (coverage)", "{:.3f}"),
    ("test_avg_gate", "Avg gate", "{:.3f}"),
    ("test_central_interval_coverage", "Quantile cov.", "{:.3f}"),
    ("train_time_s", "Train time (s)", "{:.0f}"),
]


def _fmt(series: pd.Series, f: str) -> str:
    s = series.dropna()
    if len(s) == 0:
        return "n/a"
    m = f.format(s.mean())
    return m if len(s) == 1 else f"{m} ± {f.format(s.std(ddof=1)).lstrip('+')}"


def write_report(out: Path, results: list, ref: dict, cfg, seeds) -> str:
    df = pd.DataFrame(results)
    df.to_csv(out / "results_per_run.csv", index=False)
    order = [v for v in BENCHMARKS + ABLATIONS if v in set(df.variant)]
    rows = []
    for v in order:
        g = df[df.variant == v]
        rows.append([v] + [_fmt(g[c], f) if c in g else "n/a" for c, _, f in COLUMNS])
    summ = pd.DataFrame(rows, columns=["variant"] + [n for _, n, _ in COLUMNS])
    summ.to_csv(out / "summary.csv", index=False)

    nominal = df["test_central_interval_nominal"].dropna()
    lines = [
        f"# Results ({len(seeds)} seed(s): {seeds}; {cfg.ppo.iterations} PPO iterations; test window only)",
        "",
        "Mean ± std over seeds. Test metrics use the deterministic (mean-action) policy.",
        "",
        "| " + " | ".join(summ.columns) + " |",
        "|" + "---|" * len(summ.columns),
    ]
    for _, r in summ.iterrows():
        lines.append("| " + " | ".join(str(x) for x in r.values) + " |")
    lines += [
        f"| equal_weight (reference, no learning) | {ref['sharpe']:+.3f} | {ref['cumulative_return']:+.1%} | "
        f"{ref['annual_volatility']:.1%} | {ref['max_drawdown']:.1%} | n/a | n/a | n/a | n/a | 0 |",
        "",
        f"*Quantile cov.* = share of test returns inside [z_1, z_N]; nominal = "
        f"{nominal.iloc[0]:.3f}." if len(nominal) else "",
        "*E_cal (DOCX)* is line 14 literally (floor ≈ 1/3 even when perfectly calibrated); "
        "*E_cal (coverage)* is the coverage-error variant (0 when calibrated). The gate used "
        f"`cal_error_mode = {cfg.reward.cal_error_mode}`.",
        "Standard PPO has no quantile head, so its critic columns are n/a.",
    ]
    # meta-agent decisions
    meta_lines = []
    for v in order:
        for s in seeds:
            p = out / "runs" / v / f"seed_{s}" / "meta_decisions.jsonl"
            if p.exists():
                recs = read_jsonl(p)
                acc = [r for r in recs if r["decision"] == "accept"]
                moves = ", ".join(f"{r['proposal']['delta_h']['name']}×{r['proposal']['delta_h']['factor']}"
                                  for r in acc) or "none"
                meta_lines.append(f"| {v} | {s} | {len(recs)} | {len(acc)} | {moves} |")
    if meta_lines:
        lines += ["", "## Meta-agent decisions", "",
                  "| variant | seed | calls | accepted | accepted moves |", "|---|---|---|---|---|"] + meta_lines
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    try:
        _plots(out, order, seeds)
    except Exception as e:  # plotting must never break the pipeline
        print(f"      (plotting skipped: {e})")
    return str(path)


def _plots(out: Path, order, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for v in order:
        curves = []
        for s in seeds:
            p = out / "runs" / v / f"seed_{s}" / "metrics.jsonl"
            if p.exists():
                recs = [r for r in read_jsonl(p) if "val_sharpe" in r]
                curves.append([(r["iteration"], r["val_sharpe"]) for r in recs])
        if curves:
            n = min(len(c) for c in curves)
            it = [c[0] for c in curves[0][:n]]
            m = np.mean([[c[1] for c in cv[:n]] for cv in curves], axis=0)
            axes[0].plot(it, m, label=v, lw=1.6)
    axes[0].set_title("Validation Sharpe during training (seed mean)")
    axes[0].set_xlabel("PPO iteration")
    axes[0].axhline(0, color="grey", lw=0.6)
    axes[0].legend(fontsize=7, frameon=False)

    df = pd.read_csv(out / "results_per_run.csv")
    g = df.groupby("variant")["test_sharpe"]
    means, stds = g.mean().reindex(order), g.std().reindex(order).fillna(0)
    axes[1].bar(range(len(order)), means.values, yerr=stds.values, capsize=3, color="#4c72b0")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    axes[1].axhline(0, color="grey", lw=0.6)
    axes[1].set_title("Test Sharpe (mean ± std over seeds)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=130)
    plt.close(fig)
