"""Data leakage, environment and end-to-end trainer tests (incl. meta-agent accept/rollback)."""
import copy
import tempfile
import unittest

import numpy as np

from asrl.benchmarks.metrics import max_drawdown, performance_metrics
from asrl.benchmarks.variants import VARIANTS
from asrl.config import Hyperparams, preset
from asrl.data.dataset import build_dataset, chronological_split, market_features
from asrl.env.portfolio_env import PortfolioEnv
from asrl.llm.meta_agent import HyperparamSpace, RuleBasedMetaAgent
from asrl.llm.signals import MockLLMProvider, build_signals
from asrl.ppo.trainer import Trainer
from asrl.scale_rl.turbulence import turbulence_series


def tiny_setup():
    cfg = preset("smoke")
    ds = build_dataset(cfg.data, cfg.reward.W)
    E, ell, _ = build_signals(ds.corpus, ds.n_days, MockLLMProvider(cfg.embedding_dim), cfg.signal_lookback_days)
    return cfg, ds, E, ell


class TestData(unittest.TestCase):
    def test_split_chronological_disjoint(self):
        s = chronological_split(1000, 0.6, 0.2)
        self.assertEqual(s.train, (0, 600)); self.assertEqual(s.val, (600, 800)); self.assertEqual(s.test, (800, 1000))

    def test_features_causal(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 0.01, (200, 3))
        F = market_features(y, 5, 20)
        y2 = y.copy(); y2[120:] = 1.0
        np.testing.assert_array_equal(F[:121], market_features(y2, 5, 20)[:121])   # F[d] uses y[:d]

    def test_normalisation_uses_train_only(self):
        cfg = preset("smoke")
        ds = build_dataset(cfg.data, cfg.reward.W)
        tr1 = ds.split.train[1]
        # rebuild with corrupted val/test returns: train-range features must be identical
        from asrl.data import dataset as dmod
        orig = dmod.generate_synthetic_market

        def corrupted(*a, **k):
            dates, names, y, corpus, meta = orig(*a, **k)
            y = y.copy(); y[tr1:] *= 10.0
            return dates, names, y, corpus, meta
        dmod.generate_synthetic_market = corrupted
        try:
            ds2 = build_dataset(cfg.data, cfg.reward.W)
        finally:
            dmod.generate_synthetic_market = orig
        np.testing.assert_allclose(ds.features[:tr1 + 1], ds2.features[:tr1 + 1])
        np.testing.assert_allclose(ds.turb[:tr1 + 1], ds2.turb[:tr1 + 1])


class TestEnvMetrics(unittest.TestCase):
    def test_step_return_with_costs(self):
        cfg, ds, _, _ = tiny_setup()
        env = PortfolioEnv(ds, cost=0.001)
        env.reset(np.array([100]))
        w = np.zeros((1, ds.n_assets + 1)); w[0, 0] = 0.5; w[0, -1] = 0.5
        r, y = env.step(w)
        expected = 0.5 * ds.y[100, 0] - 0.001 * (0.5 + 0.5)    # from all-cash: turnover = 1.0
        self.assertAlmostEqual(r[0], expected)
        self.assertAlmostEqual(env.hold.sum(), 1.0)

    def test_metrics(self):
        r = np.array([0.1, -0.2, 0.05])
        self.assertAlmostEqual(max_drawdown(r), (1.1 * 0.8) / 1.1 - 1.0)
        m = performance_metrics(r)
        self.assertAlmostEqual(m["cumulative_return"], 1.1 * 0.8 * 1.05 - 1)


class TestTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.ds, cls.E, cls.ell = tiny_setup()

    def make(self, name, seed=0, run_dir=None):
        v = VARIANTS[name]
        agent = RuleBasedMetaAgent(seed) if v.use_meta else None
        return Trainer(self.cfg, v, self.ds, self.E, self.ell, seed, run_dir, agent, HyperparamSpace(self.cfg.meta))

    def test_all_variants_train(self):
        for name in VARIANTS:
            with tempfile.TemporaryDirectory() as d:
                tr = self.make(name, run_dir=d)
                info = tr.train()
                m = tr.test_metrics()
                self.assertTrue(np.isfinite(m["sharpe"]), name)
                if VARIANTS[name].gate_fixed_one:
                    self.assertAlmostEqual(m["avg_gate"], 1.0)
                if VARIANTS[name].use_meta:
                    self.assertGreater(info["meta_calls"], 0)

    def test_reproducible(self):
        a, b = self.make("agentic_full", seed=3), self.make("agentic_full", seed=3)
        a.train(); b.train()
        self.assertEqual(a.val_metrics(), b.val_metrics())
        self.assertEqual([m["decision"] for m in a.memory], [m["decision"] for m in b.memory])

    def test_reward_zero_during_burn_in(self):
        tr = self.make("scale_rl")
        buf = tr.rollout()
        np.testing.assert_array_equal(buf["D"][:, :tr.Bburn], 0.0)
        self.assertTrue(np.any(buf["D"][:, tr.Bburn:] != 0.0))
        lam = buf["lam"][:, tr.Bburn:]
        self.assertTrue(np.all((lam > 0) & (lam < 1)))

    def test_meta_rollback_and_accept(self):
        tr = self.make("agentic_full")
        tr.run_iteration()
        before = copy.deepcopy(tr.state_dict())
        # force a reject: probe validation Sharpe below baseline + delta
        tr.diagnostics = lambda: {"val_sharpe": 10.0, "approx_kl": 0.0, "clip_frac": 0.0,
                                  "entropy": 1.0, "coverage_error": 0.0}
        tr.val_metrics = lambda: {"sharpe": 0.0}
        rec = tr.meta_step()
        self.assertEqual(rec["decision"], "reject_rollback")
        for net, ps in before["params"].items():
            for k, arr in ps.items():
                np.testing.assert_array_equal(tr.state_dict()["params"][net][k], arr)
        self.assertEqual(tr.h, before["h"])
        # force an accept
        tr.diagnostics = lambda: {"val_sharpe": -10.0, "approx_kl": 0.0, "clip_frac": 0.0,
                                  "entropy": 1.0, "coverage_error": 0.0}
        rec = tr.meta_step()
        self.assertEqual(rec["decision"], "accept")
        self.assertNotEqual(tr.h, before["h"])
        # reward-defining parameters untouched by the meta loop
        self.assertEqual((tr.alpha, tr.Bburn, tr.b0), (self.cfg.reward.alpha, self.cfg.reward.B, self.cfg.reward.b0))

    def test_validation_does_not_read_future(self):
        """Validation metrics must not change when data after the validation end is altered."""
        tr = self.make("agentic_full")
        tr.run_iteration()
        m1 = tr.val_metrics()
        v1 = self.ds.split.val[1]
        saved = (self.ds.y.copy(), self.ds.features.copy(), self.ds.turb.copy())
        try:
            self.ds.y[v1:] = 0.3
            self.ds.features[v1:] = 99.0
            self.ds.turb[v1:] = 1e6
            m2 = tr.val_metrics()
        finally:
            self.ds.y[:], self.ds.features[:], self.ds.turb[:] = saved
        self.assertEqual(m1, m2)


if __name__ == "__main__":
    unittest.main()
