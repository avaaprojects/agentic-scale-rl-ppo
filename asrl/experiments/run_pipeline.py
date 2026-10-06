"""One command: data -> frozen LLM signals -> train all variants x seeds -> test -> report.

    python run_pipeline.py --preset quick
    python run_pipeline.py --preset smoke            # < 1 min sanity run
    python run_pipeline.py --preset full --workers 4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from ..benchmarks.metrics import performance_metrics
from ..benchmarks.variants import ABLATIONS, BENCHMARKS, VARIANTS
from ..config import ExperimentConfig, preset
from ..data.dataset import build_dataset
from ..env.portfolio_env import portfolio_returns_fixed_weights
from ..llm.meta_agent import HyperparamSpace, make_meta_agent
from ..llm.signals import MockLLMProvider, load_or_build_signals
from ..ppo.trainer import Trainer
from ..utils import append_jsonl, set_global_seed, write_json
from .report import write_report

_G: dict = {}   # dataset and signals shared with forked workers


def _run_job(job):
    variant_name, seed = job
    cfg: ExperimentConfig = _G["cfg"]
    out: Path = _G["out"]
    variant = VARIANTS[variant_name]
    run_dir = out / "runs" / variant_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("metrics.jsonl", "meta_decisions.jsonl"):
        (run_dir / stale).unlink(missing_ok=True)
    set_global_seed(seed)
    write_json(run_dir / "config.json", {"experiment": cfg.to_dict(), "variant": variant, "seed": seed})
    meta_agent = make_meta_agent(cfg.meta.agent, seed) if variant.use_meta else None
    tr = Trainer(cfg, variant, _G["ds"], _G["E"], _G["ell"], seed=seed, run_dir=str(run_dir),
                 meta_agent=meta_agent, hp_space=HyperparamSpace(cfg.meta))
    info = tr.train()
    val = tr.val_metrics()
    test = tr.test_metrics()
    res = {"variant": variant_name, "seed": seed, **info,
           **{f"val_{k}": v for k, v in val.items()}, **{f"test_{k}": v for k, v in test.items()}}
    write_json(run_dir / "result.json", res)
    print(f"  done {variant_name:18s} seed={seed}  test Sharpe={test['sharpe']:+.3f}  "
          f"val Sharpe={val['sharpe']:+.3f}  train {info['train_time_s']:.1f}s", flush=True)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="quick", choices=["smoke", "quick", "full"])
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="default: 0 1 2 (smoke: 0)")
    p.add_argument("--variants", nargs="+", default=BENCHMARKS + ABLATIONS, choices=list(VARIANTS))
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default=None)
    p.add_argument("--cal-error-mode", choices=["docx", "coverage"], default=None)
    p.add_argument("--meta-baseline", choices=["checkpoint", "control"], default=None)
    p.add_argument("--data-csv", default=None, help="wide price CSV (date + one column per asset)")
    p.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 1))))
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    cfg = preset(args.preset)
    if args.iterations: cfg.ppo.iterations = args.iterations
    if args.optimizer: cfg.ppo.optimizer = args.optimizer
    if args.cal_error_mode: cfg.reward.cal_error_mode = args.cal_error_mode
    if args.meta_baseline: cfg.meta.baseline_mode = args.meta_baseline
    if args.data_csv:
        cfg.data.source, cfg.data.csv_path = "csv", args.data_csv
    seeds = args.seeds if args.seeds is not None else ([0] if args.preset == "smoke" else [0, 1, 2])

    out = Path(args.out or f"results/{args.preset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[1/5] config -> {out}")
    write_json(out / "config.json", {"experiment": cfg.to_dict(), "seeds": seeds, "variants": args.variants,
                                     "W": cfg.reward.W, "B": cfg.reward.B,
                                     "python": sys.version, "numpy": np.__version__,
                                     "platform": platform.platform(), "argv": sys.argv})

    print("[2/5] data (chronological split, train-only feature normalisation, Ledoit-Wolf turbulence)")
    ds = build_dataset(cfg.data, cfg.reward.W, cfg.reward.turb_window_includes_current)
    write_json(out / "dataset.json", ds.describe())
    d = ds.describe()
    print(f"      train {d['train'][:2]}  val {d['val'][:2]}  test {d['test'][:2]}")

    print("[3/5] offline LLM signals (mock provider, cached and frozen before training)")
    provider = MockLLMProvider(embedding_dim=cfg.embedding_dim)
    E, ell, cinfo = load_or_build_signals("cache", ds.corpus, ds.n_days, provider, cfg.signal_lookback_days)
    fwd_absret = np.array([np.abs(ds.y[t:t + 5]).mean() for t in range(ds.n_days)])
    sig = {**cinfo, "provider": provider.name, "provider_version": provider.version,
           "ell_mean": float(ell.mean()), "ell_std": float(ell.std()),
           "corr_ell_vs_next5d_abs_return": float(np.corrcoef(ell, fwd_absret)[0, 1])}
    write_json(out / "llm_signals.json", sig)
    print(f"      ell mean {sig['ell_mean']:.3f}; corr(ell_t, next-5d |return|) = "
          f"{sig['corr_ell_vs_next5d_abs_return']:.3f}; cache {cinfo['cache_path']}")

    print(f"[4/5] training {len(args.variants)} variants x {len(seeds)} seeds "
          f"({cfg.ppo.iterations} iterations each, {args.workers} workers)")
    _G.update(cfg=cfg, out=out, ds=ds, E=E, ell=ell)
    jobs = [(v, s) for s in seeds for v in args.variants]
    if args.workers > 1:
        with mp.get_context("fork").Pool(args.workers) as pool:
            results = pool.map(_run_job, jobs, chunksize=1)
    else:
        results = [_run_job(j) for j in jobs]
    for r in results:
        append_jsonl(out / "all_results.jsonl", r)

    # non-learning reference on the same test window
    w_eq = np.append(np.ones(ds.n_assets) / ds.n_assets, 0.0)
    a, b = ds.split.test
    ref = performance_metrics(portfolio_returns_fixed_weights(ds.y[a:b], w_eq, cfg.data.transaction_cost))
    write_json(out / "reference_equal_weight.json", ref)

    print("[5/5] report")
    summary_path = write_report(out, results, ref, cfg, seeds)
    print(f"done in {time.time() - t0:.0f}s -> {summary_path}")
    print(Path(summary_path).read_text())
    return out


if __name__ == "__main__":
    main()
