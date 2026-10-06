"""Unit tests for the SCALE-RL mathematical components (DOCX lines 8-20, 31-32)."""
import unittest

import numpy as np
from scipy.stats import norm

from asrl.models.networks import quantile_moments
from asrl.scale_rl.calibration import calibration_error, coverage_stats
from asrl.scale_rl.gate import reliability_gate, sigmoid
from asrl.scale_rl.losses import calibration_loss, pinball_loss
from asrl.scale_rl.moments import (floored_variance, gated_dsr, gated_innovations, init_moments,
                                   sharpe_fn, update_moments)
from asrl.scale_rl.turbulence import ledoit_wolf_cov, turbulence_at, turbulence_series


def num_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        x0 = x[i]
        x[i] = x0 + h; fp = f()
        x[i] = x0 - h; fm = f()
        x[i] = x0
        g[i] = (fp - fm) / (2 * h)
    return g


class TestMoments(unittest.TestCase):
    def test_init_requires_b0_ge_eps(self):
        A, B = init_moments(3, 1e-4, 1e-6)
        np.testing.assert_array_equal(A, 0.0)
        np.testing.assert_array_equal(B, 1e-4)
        with self.assertRaises(AssertionError):
            init_moments(1, 1e-7, 1e-6)

    def test_variance_floor(self):
        V = floored_variance(np.array([0.1, 0.0]), np.array([0.01, 0.5]), 1e-3)
        np.testing.assert_allclose(V, [1e-3, 0.5])

    def test_update_moments_raw_returns(self):
        A, B, V = update_moments(np.array([0.1]), np.array([0.2]), np.array([0.5]), 0.1)
        np.testing.assert_allclose(A, 0.9 * 0.1 + 0.1 * 0.5)
        np.testing.assert_allclose(B, 0.9 * 0.2 + 0.1 * 0.25)
        np.testing.assert_allclose(V, B - A ** 2)

    def test_gated_innovations_limits(self):
        r, mu, var, A, B, a = 0.03, 0.01, 4e-4, 0.002, 2e-4, 0.05
        dA1, dB1 = gated_innovations(r, mu, var, A, B, 1.0, a)        # lambda=1 -> classical DSR
        self.assertAlmostEqual(dA1, a * (r - A))
        self.assertAlmostEqual(dB1, a * (r ** 2 - B))
        dA0, dB0 = gated_innovations(r, mu, var, A, B, 0.0, a)        # lambda=0 -> critic moments
        self.assertAlmostEqual(dA0, a * (mu - A))
        self.assertAlmostEqual(dB0, a * (mu ** 2 + var - B))
        lam = 0.3                                                       # convex blend
        dA, dB = gated_innovations(r, mu, var, A, B, lam, a)
        self.assertAlmostEqual(dA, lam * dA1 + (1 - lam) * dA0)
        self.assertAlmostEqual(dB, lam * dB1 + (1 - lam) * dB0)

    def test_gated_dsr_formula_and_burn_in(self):
        A, B, eps = np.array([0.001]), np.array([2e-4]), 1e-6
        V = floored_variance(A, B, eps)
        dA, dB = np.array([3e-4]), np.array([-1e-5])
        expected = (B * dA - 0.5 * A * dB) / V ** 1.5
        np.testing.assert_allclose(gated_dsr(dA, dB, A, B, V, t=10, burn_in=10), expected)
        np.testing.assert_allclose(gated_dsr(dA, dB, A, B, V, t=9, burn_in=10), 0.0)
        # Jacobian coefficients (Eqs. 23-24)
        dSdA, dSdB = B / V ** 1.5, -A / (2 * V ** 1.5)
        np.testing.assert_allclose(expected, dSdA * dA + dSdB * dB)

    def test_dsr_first_order_sharpe_increment(self):
        """Classical DSR (lambda=1) is the first-order Sharpe increment: error O(alpha^2)."""
        rng = np.random.default_rng(0)
        for alpha in (0.02, 0.005):
            A, B = np.array([0.0005]), np.array([1.5e-4])
            errs = []
            for _ in range(200):
                r = rng.normal(0.0005, 0.012, 1)
                V = floored_variance(A, B, 1e-8)
                dA, dB = gated_innovations(r, 0, 0, A, B, 1.0, alpha)
                D = gated_dsr(dA, dB, A, B, V, 1, 0)
                A2, B2, _ = update_moments(A, B, r, alpha)
                errs.append(abs(sharpe_fn(A2, B2) - sharpe_fn(A, B) - D)[0])
                A, B = A2, B2
            if alpha == 0.02:
                e_big = np.mean(errs)
            else:
                e_small = np.mean(errs)
        # quartering alpha should cut the second-order residual by roughly 16x (allow >= 6x)
        self.assertGreater(e_big / e_small, 6.0)

    def test_theorem2_telescoping(self):
        """Sum_{t>=B} D_t ~= S_T - S_B (Theorem 2, lambda=1) up to O(alpha^2) per step."""
        rng = np.random.default_rng(1)
        alpha, T, Bb = 0.01, 600, 100
        A, B = init_moments(1, 1.4e-4, 1e-8)
        total = 0.0
        S_B = None
        for t in range(T):
            if t == Bb:
                S_B = sharpe_fn(A, B)[0]
            r = rng.normal(0.0006, 0.012, 1)
            V = floored_variance(A, B, 1e-8)
            dA, dB = gated_innovations(r, 0, 0, A, B, 1.0, alpha)
            total += gated_dsr(dA, dB, A, B, V, t, Bb)[0]
            A, B, _ = update_moments(A, B, r, alpha)
        S_T = sharpe_fn(A, B)[0]
        self.assertAlmostEqual(total, S_T - S_B, delta=0.05 * max(1.0, abs(S_T - S_B)))


class TestCalibration(unittest.TestCase):
    taus = (2 * np.arange(1, 9) - 1) / 16

    def test_t0_is_one(self):
        np.testing.assert_array_equal(calibration_error(np.zeros((2, 5)), np.zeros((2, 5, 8)),
                                                        self.taus, 0, 10), [1.0, 1.0])

    def test_docx_formula_small_example(self):
        taus = np.array([0.25, 0.75])
        r = np.array([[0.0, 1.0, 5.0]])
        z = np.array([[[0.5, 2.0], [0.5, 2.0], [0.0, 0.0]]])
        # t=2, W=4: j in {0,1}; j0: r=0 <= both -> |1-.25|,|1-.75| mean .5 ; j1: r=1 -> |0-.25|,|1-.75| -> .25
        e = calibration_error(r, z, taus, 2, 4, "docx")
        np.testing.assert_allclose(e, (0.5 + 0.25) / 4)
        cov = calibration_error(r, z, taus, 2, 4, "coverage")
        np.testing.assert_allclose(cov, (abs(0.5 - 0.25) + abs(1.0 - 0.75)) / 2)

    def test_uses_only_past(self):
        rng = np.random.default_rng(0)
        r = rng.normal(size=(1, 30)); z = rng.normal(size=(1, 30, 8))
        e1 = calibration_error(r, z, self.taus, 20, 10)
        r[:, 20:] = 99.0; z[:, 20:] = -99.0
        np.testing.assert_array_equal(e1, calibration_error(r, z, self.taus, 20, 10))

    def test_perfect_calibration_floor(self):
        """Documents the DOCX-literal floor: calibrated critic -> mean 2 tau (1-tau), coverage -> 0."""
        rng = np.random.default_rng(0)
        n, W = 20000, 20000
        r = rng.normal(size=(1, n))
        z = np.broadcast_to(norm.ppf(self.taus), (1, n, 8)).copy()
        e_doc = calibration_error(r, z, self.taus, n, W, "docx")[0]
        e_cov = calibration_error(r, z, self.taus, n, W, "coverage")[0]
        self.assertAlmostEqual(e_doc, float(np.mean(2 * self.taus * (1 - self.taus))), delta=0.01)
        self.assertLess(e_cov, 0.01)

    def test_coverage_stats(self):
        rng = np.random.default_rng(0)
        r = rng.normal(size=50000)
        z = np.broadcast_to(norm.ppf(self.taus), (50000, 8))
        cs = coverage_stats(r, z, self.taus)
        self.assertLess(cs["coverage_error"], 0.01)
        self.assertAlmostEqual(cs["central_interval_coverage"], cs["central_interval_nominal"], delta=0.01)


class TestGateAndTurbulence(unittest.TestCase):
    def test_gate(self):
        lam = reliability_gate(0.3, 8.0, 0.4, -2.0, 2.0, 0.1, 2.0)
        self.assertAlmostEqual(float(lam), 1 / (1 + np.exp(-(-2 + 0.6 + 0.8 + 0.8))))
        self.assertTrue(np.all((sigmoid(np.array([-800.0, 0.0, 800.0])) >= 0)))
        base = reliability_gate(0.3, 8.0, 0.4, -2, 2, 0.1, 2)
        for kw in ({"E": 0.5}, {"T": 12.0}, {"L": 0.9}):
            v = reliability_gate(kw.get("E", 0.3), kw.get("T", 8.0), kw.get("L", 0.4), -2, 2, 0.1, 2)
            self.assertGreater(float(v), float(base))

    def test_turbulence_matches_manual_ledoit_wolf(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 0.01, size=(120, 4))
        W, t = 50, 100
        win = y[t - 1 - W:t - 1]
        d = y[t - 1] - win.mean(0)
        expect = d @ np.linalg.inv(ledoit_wolf_cov(win)) @ d
        self.assertAlmostEqual(turbulence_at(y, t, W), expect, places=8)

    def test_turbulence_causal(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 0.01, size=(150, 3))
        turb, valid = turbulence_series(y, 40)
        y2 = y.copy(); y2[100:] = 5.0
        turb2, _ = turbulence_series(y2, 40)
        np.testing.assert_allclose(turb[:101], turb2[:101])   # Turb_t uses y_{<t} only
        self.assertFalse(valid[40]); self.assertTrue(valid[41])


class TestCriticLosses(unittest.TestCase):
    def test_quantile_moments(self):
        z = np.array([[1.0, 2.0, 3.0, 6.0]])
        mu, var = quantile_moments(z)
        np.testing.assert_allclose(mu, 3.0)
        np.testing.assert_allclose(var, np.var(z))

    def test_pinball_value_and_grad(self):
        rng = np.random.default_rng(0)
        taus = (2 * np.arange(1, 6) - 1) / 10
        z, r = rng.normal(size=(7, 5)), rng.normal(size=7)
        loss, dz = pinball_loss(z, r, taus)
        manual = np.mean([[(taus[i] - (r[m] < z[m, i])) * (r[m] - z[m, i]) for i in range(5)] for m in range(7)])
        self.assertAlmostEqual(loss, manual)
        np.testing.assert_allclose(dz, num_grad(lambda: pinball_loss(z, r, taus)[0], z), atol=1e-6)
        self.assertGreaterEqual(loss, 0.0)

    def test_pinball_minimiser_is_quantile(self):
        rng = np.random.default_rng(0)
        r = rng.standard_normal(4000)
        taus = np.array([0.1, 0.5, 0.9])
        z = np.zeros((1, 3))
        for _ in range(4000):
            _, dz = pinball_loss(np.repeat(z, len(r), 0), r, taus)
            z -= 2.0 * dz.sum(0, keepdims=True)
        np.testing.assert_allclose(z[0], np.quantile(r, taus), atol=0.05)

    def test_calibration_loss_value_and_grad(self):
        rng = np.random.default_rng(2)
        z, A, V = rng.normal(size=(6, 4)), rng.normal(size=6), rng.random(6)
        loss, dz = calibration_loss(z, A, V)
        mu, var = quantile_moments(z)
        self.assertAlmostEqual(loss, float(np.mean((mu - A) ** 2 + (var - V) ** 2)))
        np.testing.assert_allclose(dz, num_grad(lambda: calibration_loss(z, A, V)[0], z), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
