"""Agentic SCALE-RL-PPO trainer: DOCX Algorithm 1, lines 1 and 5-46.

Line numbers in comments refer to the DOCX algorithm. Variant switches
(VariantConfig) turn components off to obtain the baselines and ablations.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..benchmarks.metrics import performance_metrics
from ..config import ExperimentConfig, Hyperparams, VariantConfig
from ..data.dataset import MarketDataset
from ..env.portfolio_env import PortfolioEnv
from ..models.networks import GaussianActor, QuantileNet, ValueNet, quantile_moments, softmax
from ..models.optim import make_optimizer
from ..scale_rl.calibration import calibration_error, coverage_stats
from ..scale_rl.gate import reliability_gate
from ..scale_rl.losses import calibration_loss, pinball_loss
from ..scale_rl.moments import (floored_variance, gated_dsr, gated_innovations, init_moments,
                                update_moments)
from ..utils import append_jsonl, write_json
from .core import compute_gae, normalise, ppo_clip_loss

A_SCALE, B_SCALE = 0.01, 1e-4   # fixed linear rescaling of A, B when fed to the networks


class Trainer:
    def __init__(self, cfg: ExperimentConfig, variant: VariantConfig, ds: MarketDataset,
                 E: Optional[np.ndarray], ell: Optional[np.ndarray], seed: int,
                 run_dir: Optional[str] = None, meta_agent=None, hp_space=None):
        cfg.reward.validate()
        self.cfg, self.v, self.ds = cfg, variant, ds
        self.E, self.ell = E, ell
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.run_dir = Path(run_dir) if run_dir else None
        self.meta_agent, self.hp_space = meta_agent, hp_space
        rc, pc = cfg.reward, cfg.ppo
        self.alpha, self.W, self.Bburn, self.eps, self.b0 = rc.alpha, rc.W, rc.B, rc.eps, rc.b0

        self.env = PortfolioEnv(ds, cfg.data.transaction_cost)
        s_dim, a_dim = self.env.state_dim, self.env.action_dim
        aug = s_dim
        if variant.use_llm_embedding:
            assert E is not None
            aug += E.shape[1]
        if variant.use_llm_risk:
            assert ell is not None
            aug += 1
        if variant.use_ab_state:
            aug += 2
        self.aug_dim = aug

        # Line 1: initialise actor, value head, quantile head; h <- h0; memory <- {}
        self.actor = GaussianActor(aug, a_dim, pc.hidden, pc.n_hidden_layers, self.rng,
                                   pc.log_std_init, pc.log_std_bounds)
        self.value = ValueNet(aug, pc.hidden, pc.n_hidden_layers, self.rng)
        self.qnet = QuantileNet(s_dim + a_dim, pc.n_quantiles, pc.hidden, pc.n_hidden_layers,
                                self.rng, pc.quantile_out_scale) if variant.use_quantile else None
        self.opt_theta = make_optimizer(pc.optimizer, self.actor.params)
        self.opt_psi = make_optimizer(pc.optimizer, self.value.params)
        self.opt_phi = make_optimizer(pc.optimizer, self.qnet.params) if self.qnet else None
        self.h: Hyperparams = cfg.h0.copy()
        self.memory: list = []
        self.iteration = 0              # main-loop iterations k
        self.total_updates = 0          # includes probe iterations
        self.last_stats: dict = {}
        self._in_rollout = False        # guard: the meta-agent must never run inside a rollout

    # ------------------------------------------------------------------ state
    def aug_state(self, s, d, A_prev, B_prev):
        """Line 11: s~_t = (s_t, e_t, ell_t, A_{t-1}, B_{t-1}) (components per variant)."""
        parts = [s]
        if self.v.use_llm_embedding:
            parts.append(self.E[d])
        if self.v.use_llm_risk:
            parts.append(self.ell[d][:, None])
        if self.v.use_ab_state:
            parts.append(np.stack([A_prev / A_SCALE, B_prev / B_SCALE], axis=1))
        return np.concatenate(parts, axis=1)

    def _gate(self, E_cal, d):
        rc = self.cfg.reward
        if self.v.gate_fixed_one:
            return np.ones_like(E_cal)
        ell_t = self.ell[d] if self.v.use_llm_risk else np.zeros(len(d))
        w3 = rc.w3 if self.v.use_llm_risk else 0.0
        return reliability_gate(E_cal, self.ds.turb[d], ell_t, rc.w0, rc.w1, rc.w2, w3)

    # ------------------------------------------------------------------ rollout
    def rollout(self):
        """Lines 6-23. Critic (phi) is not updated during the rollout, so phi_bar = phi."""
        pc = self.cfg.ppo
        n, T = pc.n_env, pc.T
        lo = self.ds.first_valid
        hi = self.ds.split.train[1] - T
        assert hi >= lo, "training range shorter than episode length T"
        starts = self.rng.integers(lo, hi + 1, size=n)
        self._in_rollout = True
        s = self.env.reset(starts)
        A, B = init_moments(n, self.b0, self.eps)                         # line 8
        N = pc.n_quantiles
        buf = {k: np.zeros((n, T)) for k in ("logp", "D", "r", "A", "V", "val", "lam", "ecal")}
        buf["sa"] = np.zeros((n, T, self.aug_dim))
        buf["xc"] = np.zeros((n, T, self.env.state_dim + self.env.action_dim))
        buf["u"] = np.zeros((n, T, self.env.action_dim))
        Zh = np.zeros((n, T, N))
        Rh = np.zeros((n, T))
        for t in range(T):                                                # line 9
            d = self.env.d
            Vfl = floored_variance(A, B, self.eps)                        # line 10
            sa = self.aug_state(s, d, A, B)                               # line 11
            u, logp, _ = self.actor.sample(sa, self.rng)                  # line 12
            w = softmax(u)
            val, _ = self.value.forward(sa)
            r, _y = self.env.step(w)
            if self.qnet is not None:
                xc = np.concatenate([s, w], axis=1)
                z, _ = self.qnet.forward(xc)                              # line 13
                mu_hat, var_hat = quantile_moments(z)
                E_cal = calibration_error(Rh, Zh, self.qnet.taus, t, self.W,
                                          self.cfg.reward.cal_error_mode)  # line 14 (j < t)
                lam = self._gate(E_cal, d)                                # lines 15-16
                Zh[:, t], Rh[:, t] = z, r
                buf["xc"][:, t] = xc
            else:
                mu_hat = var_hat = np.zeros(n)
                E_cal, lam = np.zeros(n), np.ones(n)
            if self.v.reward == "gated_dsr":
                dA, dB = gated_innovations(r, mu_hat, var_hat, A, B, lam, self.alpha)  # 17-18
                D = gated_dsr(dA, dB, A, B, Vfl, t, self.Bburn)           # line 19
            else:
                D = r                                                     # standard PPO reward
            A, B, Vt = update_moments(A, B, r, self.alpha)                # line 20
            for k, x in (("logp", logp), ("D", D), ("r", r), ("A", A), ("V", Vt),
                         ("val", val), ("lam", lam), ("ecal", E_cal)):
                buf[k][:, t] = x                                          # line 21
            buf["sa"][:, t], buf["u"][:, t] = sa, u
            if t < T - 1:
                s = self.env.obs()
        self._in_rollout = False
        return buf

    # ------------------------------------------------------------------ update
    def ppo_update(self, buf):
        """Lines 24-36."""
        pc, h = self.cfg.ppo, self.h
        adv, G = compute_gae(buf["D"], buf["val"], pc.gamma, pc.lambda_gae)   # lines 24-25
        adv = normalise(adv)
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        sa, u, logp_old = flat(buf["sa"]), flat(buf["u"]), flat(buf["logp"])
        adv, G = adv.reshape(-1), G.reshape(-1)
        r, A_t, V_t, xc = flat(buf["r"]), flat(buf["A"]), flat(buf["V"]), flat(buf["xc"])
        nb = len(adv)
        logs = {k: [] for k in ("L_clip", "L_V", "L_pin", "L_cal", "L", "clip_frac", "approx_kl")}
        for _ in range(pc.epochs):                                           # line 26
            perm = self.rng.permutation(nb)
            for i in range(0, nb, pc.minibatch):                             # line 27
                mb = perm[i:i + pc.minibatch]
                m = len(mb)
                # actor: L_clip - c_e H (lines 28-29, 33)
                mu, cache = self.actor.mean(sa[mb])
                logp_new = self.actor.log_prob(u[mb], mu)
                l_clip, dlogp, st = ppo_clip_loss(logp_new, logp_old[mb], adv[mb], h.eps_clip)
                g = self.actor.grads(cache, u[mb], mu, dlogp, d_entropy=-h.c_e)
                self.opt_theta.step(g, h.eta_theta)                          # line 34
                self.actor.clamp()
                # value: c_v L_V (line 30)
                v, vc = self.value.forward(sa[mb])
                l_v = float(((v - G[mb]) ** 2).mean())
                self.opt_psi.step(self.value.grads(vc, h.c_v * 2.0 * (v - G[mb]) / m), h.eta_psi)
                # quantile head: c_q (L_pin + beta L_cal) (lines 31-32)
                l_pin = l_cal = 0.0
                if self.qnet is not None:
                    z, qc = self.qnet.forward(xc[mb])
                    l_pin, dzp = pinball_loss(z, r[mb], self.qnet.taus)
                    l_cal, dzc = calibration_loss(z, A_t[mb], V_t[mb])
                    self.opt_phi.step(self.qnet.grads(qc, h.c_q * (dzp + h.beta * dzc)), h.eta_phi)
                H = self.actor.entropy()
                L = l_clip + h.c_v * l_v + h.c_q * (l_pin + h.beta * l_cal) - h.c_e * H   # line 33
                for k, x in (("L_clip", l_clip), ("L_V", l_v), ("L_pin", l_pin), ("L_cal", l_cal),
                             ("L", L), ("clip_frac", st["clip_frac"]), ("approx_kl", st["approx_kl"])):
                    logs[k].append(x)
        out = {k: float(np.mean(x)) for k, x in logs.items()}
        out["entropy"] = self.actor.entropy()
        return out

    def run_iteration(self) -> dict:
        """One pass of lines 6-37 under the current h."""
        t0 = time.time()
        buf = self.rollout()
        stats = self.ppo_update(buf)
        stats.update({
            "mean_reward": float(buf["D"].mean()), "mean_r": float(buf["r"].mean()),
            "mean_gate": float(buf["lam"][:, self.Bburn:].mean()) if self.qnet else 1.0,
            "mean_ecal": float(buf["ecal"][:, self.Bburn:].mean()) if self.qnet else None,
            "iter_time_s": time.time() - t0,
        })
        self.total_updates += 1
        self.last_stats = stats
        return stats                                                         # line 37: buffer dropped

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, start: int, end: int, warmup: Optional[int] = None) -> dict:
        """Deterministic policy (mean action) run sequentially over [start, end).
        Warms up on PRIOR data only (moments, E_cal history, holdings); scores [start, end)."""
        warmup = max(self.Bburn, self.W) if warmup is None else warmup
        s0 = max(self.ds.first_valid, start - warmup)
        env = PortfolioEnv(self.ds, self.cfg.data.transaction_cost)
        s = env.reset(np.array([s0]))
        A, B = init_moments(1, self.b0, self.eps)
        n_steps = end - s0
        N = self.cfg.ppo.n_quantiles
        Zh, Rh = np.zeros((1, n_steps, N)), np.zeros((1, n_steps))
        rec = {k: [] for k in ("r", "lam", "ecal_docx", "ecal_cov", "w")}
        zs = []
        for t in range(n_steps):
            d = env.d
            sa = self.aug_state(s, d, A, B)
            mu, _ = self.actor.mean(sa)
            w = softmax(mu)
            r, _ = env.step(w)
            if self.qnet is not None:
                z, _ = self.qnet.forward(np.concatenate([s, w], axis=1))
                e_doc = calibration_error(Rh, Zh, self.qnet.taus, t, self.W, "docx")
                e_cov = calibration_error(Rh, Zh, self.qnet.taus, t, self.W, "coverage")
                e_used = e_doc if self.cfg.reward.cal_error_mode == "docx" else e_cov
                lam = self._gate(e_used, d)
                Zh[:, t], Rh[:, t] = z, r
            else:
                z = None
                e_doc = e_cov = lam = np.array([np.nan])
            A, B, _ = update_moments(A, B, r, self.alpha)
            if d[0] >= start:
                rec["r"].append(r[0]); rec["lam"].append(lam[0])
                rec["ecal_docx"].append(e_doc[0]); rec["ecal_cov"].append(e_cov[0])
                rec["w"].append(w[0])
                if z is not None:
                    zs.append(z[0])
            if t < n_steps - 1:
                s = env.obs()
        r = np.array(rec["r"])
        out = performance_metrics(r)
        w = np.array(rec["w"])
        out["avg_cash_weight"] = float(w[:, -1].mean())
        out["avg_turnover"] = float(np.abs(np.diff(w, axis=0)).sum(1).mean()) if len(w) > 1 else 0.0
        if self.qnet is not None:
            out.update({
                "calibration_error_docx": float(np.mean(rec["ecal_docx"])),
                "calibration_error_coverage": float(np.mean(rec["ecal_cov"])),
                "avg_gate": float(np.mean(rec["lam"])),
            })
            out.update(coverage_stats(r, np.array(zs), self.qnet.taus))
        return out

    def val_metrics(self) -> dict:
        return self.evaluate(*self.ds.split.val)

    def test_metrics(self) -> dict:
        return self.evaluate(*self.ds.split.test)

    # ------------------------------------------------------------------ checkpoints
    def state_dict(self) -> dict:
        nets = {"actor": self.actor.params, "value": self.value.params}
        opts = {"actor": self.opt_theta, "value": self.opt_psi}
        if self.qnet is not None:
            nets["qnet"], opts["qnet"] = self.qnet.params, self.opt_phi
        return {
            "params": {k: copy.deepcopy(p) for k, p in nets.items()},
            "opt": {k: o.state_dict() for k, o in opts.items()},
            "h": self.h.copy(),
            "total_updates": self.total_updates,
        }

    def load_state_dict(self, sd: dict) -> None:
        nets = {"actor": self.actor.params, "value": self.value.params}
        opts = {"actor": self.opt_theta, "value": self.opt_psi}
        if self.qnet is not None:
            nets["qnet"], opts["qnet"] = self.qnet.params, self.opt_phi
        for k, p in nets.items():
            for name, arr in sd["params"][k].items():
                p[name][...] = arr                   # in place: optimisers keep references
            opts[k].load_state_dict(sd["opt"][k])
        self.h = sd["h"].copy()

    def save_checkpoint(self, path) -> None:
        sd = self.state_dict()
        flat = {f"{net}/{name}": arr for net, ps in sd["params"].items() for name, arr in ps.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **flat)
        write_json(str(path).replace(".npz", "_h.json"),
                   {"h": sd["h"].as_dict(), "iteration": self.iteration, "total_updates": self.total_updates})

    def load_checkpoint(self, path) -> None:
        z = np.load(path)
        nets = {"actor": self.actor.params, "value": self.value.params}
        if self.qnet is not None:
            nets["qnet"] = self.qnet.params
        for key in z.files:
            net, name = key.split("/", 1)
            nets[net][name][...] = z[key]

    # ------------------------------------------------------------------ meta-agent
    def diagnostics(self) -> dict:
        """Line 39: S_k = Diag(pi_theta; V). KL, clip fraction and entropy come from the
        most recent PPO update (they are training quantities); the rest from validation."""
        vm = self.val_metrics()
        ecal_key = "calibration_error_docx" if self.cfg.reward.cal_error_mode == "docx" \
            else "calibration_error_coverage"
        return {
            "val_sharpe": vm["sharpe"],
            "mean_calibration_error": vm.get(ecal_key),
            "mean_gate": vm.get("avg_gate"),
            "coverage_error": vm.get("coverage_error"),
            "central_interval_coverage": vm.get("central_interval_coverage"),
            "approx_kl": self.last_stats.get("approx_kl"),
            "clip_frac": self.last_stats.get("clip_frac"),
            "entropy": self.last_stats.get("entropy"),
        }

    def _probe(self, h_new: Hyperparams, n_iter: int) -> float:
        self.h = h_new.copy()
        for _ in range(n_iter):
            self.run_iteration()
        return self.val_metrics()["sharpe"]

    def meta_step(self) -> dict:
        """Lines 39-43. Called only between iterations, never inside a rollout."""
        mc = self.cfg.meta
        assert not self._in_rollout, "meta-agent called inside a rollout"
        diag = self.diagnostics()                                            # line 39
        proposal = self.meta_agent.propose(diag, self.memory, self.h, self.hp_space)  # line 40
        record = {"iteration": self.iteration, "diagnostics": diag, "h_before": self.h.as_dict()}
        if proposal is None:
            record.update({"proposal": None, "decision": "no_proposal"})
            self.memory.append(record)
            return record
        dh = proposal["delta_h"]
        h_old = self.h.copy()
        h_new = self.hp_space.apply(h_old, dh)                               # h' = h (+) dh
        ckpt = self.state_dict()                                             # current checkpoint
        if mc.baseline_mode == "control":
            s_base = self._probe(h_old, mc.I_probe)
            self.load_state_dict(ckpt)
        else:
            s_base = diag["val_sharpe"]                                      # S_V(h) at checkpoint
        s_new = self._probe(h_new, mc.I_probe)                               # line 41: probe
        accept = s_new >= s_base + mc.delta                                  # line 42
        if accept:
            self.h = h_new                                                   # keep probe checkpoint
        else:
            self.load_state_dict(ckpt)                                       # restore checkpoint
        record.update({
            "proposal": {"delta_h": dh, "rationale": proposal.get("rationale", "")},
            "h_candidate": h_new.as_dict(), "S_V_baseline": s_base, "S_V_probe": s_new,
            "delta": mc.delta, "decision": "accept" if accept else "reject_rollback",
            "h_after": self.h.as_dict(), "baseline_mode": mc.baseline_mode,
        })
        self.memory.append(record)                                           # line 43
        return record

    # ------------------------------------------------------------------ main loop
    def train(self) -> dict:
        pc, mc = self.cfg.ppo, self.cfg.meta
        t_start = time.time()
        for k in range(1, pc.iterations + 1):                                # line 5
            self.iteration = k
            stats = self.run_iteration()
            stats["iteration"] = k
            stats["h"] = self.h.as_dict()
            if k % pc.eval_every == 0 or k == pc.iterations:
                vm = self.val_metrics()
                stats["val_sharpe"] = vm["sharpe"]
                stats["val_avg_gate"] = vm.get("avg_gate")
                stats["val_calibration_error_docx"] = vm.get("calibration_error_docx")
                stats["val_coverage_error"] = vm.get("coverage_error")
            if self.run_dir:
                append_jsonl(self.run_dir / "metrics.jsonl", stats)
            if self.v.use_meta and self.meta_agent is not None and k % mc.K_meta == 0:  # line 38
                rec = self.meta_step()
                if self.run_dir:
                    append_jsonl(self.run_dir / "meta_decisions.jsonl", rec)
        train_time = time.time() - t_start
        if self.run_dir:
            self.save_checkpoint(self.run_dir / "checkpoint_final.npz")
        return {"train_time_s": train_time, "total_ppo_iterations": self.total_updates,
                "final_h": self.h.as_dict(),
                "meta_accepts": sum(1 for m in self.memory if m.get("decision") == "accept"),
                "meta_calls": len(self.memory)}
