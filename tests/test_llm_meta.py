"""Tests for the offline LLM signal layer and the meta-agent (DOCX lines 2-4, 38-44)."""
import tempfile
import unittest

import numpy as np

from asrl.config import Hyperparams, MetaConfig
from asrl.data.synthetic import TextItem
from asrl.llm.meta_agent import HyperparamSpace, LLMMetaAgent, RuleBasedMetaAgent
from asrl.llm.signals import (MockLLMProvider, build_signals, g_verbalised_probability,
                              load_or_build_signals)

CORPUS = [TextItem(0, "markets steady and calm"), TextItem(3, "central bank policy meeting scheduled in 4 days"),
          TextItem(5, "panic selling hits equities as volatility jumps"), TextItem(9, "stocks edge higher")]


class TestSignals(unittest.TestCase):
    def test_g(self):
        self.assertEqual(g_verbalised_probability("probability: 0.37"), 0.37)
        self.assertEqual(g_verbalised_probability("I'd say 40%"), 0.40)
        self.assertEqual(g_verbalised_probability("probability = 1.7"), 0.017)   # >1 read as percent
        self.assertEqual(g_verbalised_probability("no idea"), 0.5)

    def test_strictly_before_t(self):
        """Changing or adding text stamped at day >= t must not change the signal at t."""
        prov = MockLLMProvider(8)
        E, ell, _ = build_signals(CORPUS, 12, prov, lookback=10)
        future = CORPUS + [TextItem(6, "panic panic recession fears"), TextItem(11, "volatility selloff")]
        E2, ell2, _ = build_signals(future, 12, prov, lookback=10)
        np.testing.assert_array_equal(E[:7], E2[:7])
        np.testing.assert_array_equal(ell[:7], ell2[:7])
        self.assertFalse(np.allclose(ell[7], ell2[7]))       # the day-6 item is visible from t=7
        self.assertTrue(np.all((ell >= 0) & (ell <= 1)))
        np.testing.assert_array_equal(E[0], 0.0)             # nothing published before t=0

    def test_announcement_raises_risk_before_event(self):
        E, ell, _ = build_signals(CORPUS[:2], 9, MockLLMProvider(8), lookback=10)
        self.assertGreater(ell[4], ell[2])                   # announcement (day 3) visible from t=4

    def test_cache_frozen_and_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            E1, l1, i1 = load_or_build_signals(d, CORPUS, 12, MockLLMProvider(8), 10)
            E2, l2, i2 = load_or_build_signals(d, CORPUS, 12, MockLLMProvider(8), 10)
            self.assertFalse(i1["from_cache"]); self.assertTrue(i2["from_cache"])
            np.testing.assert_array_equal(E1, E2); np.testing.assert_array_equal(l1, l2)
            _, _, i3 = load_or_build_signals(d, CORPUS[:-1], 12, MockLLMProvider(8), 10)
            self.assertNotEqual(i1["cache_key"], i3["cache_key"])


class TestHyperparamSpace(unittest.TestCase):
    def setUp(self):
        self.space = HyperparamSpace(MetaConfig())
        self.h = Hyperparams()

    def test_reward_params_rejected(self):
        for name in ("alpha", "burn_in", "b0", "w0", "w1", "w2", "w3", "eps"):
            with self.assertRaises(ValueError):
                self.space.apply(self.h, {"name": name, "factor": 2.0})

    def test_bounds_steps_and_ordering(self):
        with self.assertRaises(ValueError):
            self.space.apply(self.h, {"name": "beta", "factor": 3.0})          # not an allowed step
        h = self.h.copy(); h.eps_clip = 0.4
        with self.assertRaises(ValueError):
            self.space.apply(h, {"name": "eps_clip", "factor": 1.5})            # out of bounds
        h = Hyperparams(eta_theta=1e-3, eta_psi=1e-3, eta_phi=1e-3)
        with self.assertRaises(ValueError):
            self.space.apply(h, {"name": "eta_theta", "factor": 2.0})           # breaks two-timescale order
        for m in self.space.valid_moves(self.h):
            h2 = self.space.apply(self.h, m)
            self.assertTrue(HyperparamSpace.ordering_ok(h2))
            changed = [k for k in self.h.as_dict() if getattr(h2, k) != getattr(self.h, k)]
            self.assertEqual(changed, [m["name"]])                              # exactly ONE change

    def test_rule_agent_respects_tabu(self):
        agent = RuleBasedMetaAgent(seed=0)
        diag = {"approx_kl": 0.1, "clip_frac": 0.5, "val_sharpe": 0.1, "coverage_error": 0.01}
        p1 = agent.propose(diag, [], self.h, self.space)
        self.assertEqual(p1["delta_h"], {"name": "eta_theta", "factor": 0.5})
        mem = [{"decision": "reject_rollback", "proposal": p1, "h_before": self.h.as_dict()}]
        p2 = agent.propose(diag, mem, self.h, self.space)
        self.assertNotEqual(p2["delta_h"], p1["delta_h"])

    def test_llm_agent_parsing(self):
        agent = LLMMetaAgent(lambda prompt: 'Sure: {"move_id": 0, "rationale": "test"}')
        p = agent.propose({"val_sharpe": 0.0}, [], self.h, self.space)
        self.assertEqual(p["delta_h"], self.space.valid_moves(self.h)[0])
        self.assertIn("Allowed moves", agent.last_prompt)
        self.assertNotIn("alpha", agent.last_prompt.split("Allowed moves")[1])
        bad = LLMMetaAgent(lambda prompt: "change alpha to 0.5")
        self.assertIsNone(bad.propose({}, [], self.h, self.space))
        oob = LLMMetaAgent(lambda prompt: '{"move_id": 999}')
        self.assertIsNone(oob.propose({}, [], self.h, self.space))


if __name__ == "__main__":
    unittest.main()
