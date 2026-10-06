"""Gradient checks for the numpy networks and PPO components (DOCX lines 24-34)."""
import unittest

import numpy as np
from scipy.stats import multivariate_normal

from asrl.models.mlp import MLP
from asrl.models.networks import GaussianActor, QuantileNet, ValueNet
from asrl.models.optim import SGD
from asrl.ppo.core import compute_gae, ppo_clip_loss
from tests.test_scale_rl import num_grad


class TestMLP(unittest.TestCase):
    def test_backward_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        net = MLP([5, 7, 6, 3], rng, out_gain=1.0)
        x = rng.normal(size=(4, 5))
        c = rng.normal(size=(4, 3))
        f = lambda: float((net.forward(x)[0] * c).sum())
        _, cache = net.forward(x)
        g = net.backward(cache, c)
        for k, p in net.params.items():
            np.testing.assert_allclose(g[k], num_grad(f, p), rtol=1e-5, atol=1e-7, err_msg=k)


class TestActor(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.actor = GaussianActor(4, 3, 8, 1, self.rng, log_std_init=-0.3)
        self.actor.params["log_std"][:] = [-0.3, 0.1, -0.7]

    def test_log_prob_matches_scipy(self):
        s = self.rng.normal(size=(5, 4))
        u, logp, mu = self.actor.sample(s, self.rng)
        cov = np.diag(np.exp(2 * self.actor.params["log_std"]))
        ref = [multivariate_normal(mu[i], cov).logpdf(u[i]) for i in range(5)]
        np.testing.assert_allclose(logp, ref, rtol=1e-10)

    def test_entropy_matches_scipy(self):
        cov = np.diag(np.exp(2 * self.actor.params["log_std"]))
        self.assertAlmostEqual(self.actor.entropy(), multivariate_normal(np.zeros(3), cov).entropy())

    def test_grads(self):
        s = self.rng.normal(size=(6, 4))
        u, _, _ = self.actor.sample(s, self.rng)
        c = self.rng.normal(size=6)
        ce = 0.37

        def f():
            mu, _ = self.actor.mean(s)
            return float((c * self.actor.log_prob(u, mu)).sum() + ce * self.actor.entropy())
        mu, cache = self.actor.mean(s)
        g = self.actor.grads(cache, u, mu, c, d_entropy=ce)
        for k, p in self.actor.params.items():
            np.testing.assert_allclose(g[k], num_grad(f, p), rtol=1e-5, atol=1e-7, err_msg=k)


class TestHeads(unittest.TestCase):
    def test_value_and_quantile_grads(self):
        rng = np.random.default_rng(1)
        v = ValueNet(4, 6, 1, rng)
        q = QuantileNet(4, 5, 6, 1, rng, out_scale=0.01)
        x = rng.normal(size=(3, 4))
        cv, cq = rng.normal(size=3), rng.normal(size=(3, 5))
        _, vc = v.forward(x); gv = v.grads(vc, cv)
        _, qc = q.forward(x); gq = q.grads(qc, cq)
        for k, p in v.params.items():
            np.testing.assert_allclose(gv[k], num_grad(lambda: float((v.forward(x)[0] * cv).sum()), p),
                                       rtol=1e-5, atol=1e-8)
        for k, p in q.params.items():
            np.testing.assert_allclose(gq[k], num_grad(lambda: float((q.forward(x)[0] * cq).sum()), p),
                                       rtol=1e-5, atol=1e-9)
        np.testing.assert_allclose(q.taus, [0.1, 0.3, 0.5, 0.7, 0.9])     # tau_i = (2i-1)/(2N)


class TestPPO(unittest.TestCase):
    def test_gae_matches_definition(self):
        rng = np.random.default_rng(0)
        D, V = rng.normal(size=(2, 7)), rng.normal(size=(2, 7))
        g, lam = 0.97, 0.9
        adv, G = compute_gae(D, V, g, lam)
        Vn = np.concatenate([V[:, 1:], np.zeros((2, 1))], 1)          # V(s_T) = 0
        xi = D + g * Vn - V
        for e in range(2):
            for t in range(7):
                ref = sum((g * lam) ** l * xi[e, t + l] for l in range(7 - t))
                self.assertAlmostEqual(adv[e, t], ref)
        np.testing.assert_allclose(G, adv + V)

    def test_clip_loss_value_and_grad(self):
        rng = np.random.default_rng(0)
        old = rng.normal(size=40)
        new = old + rng.normal(0, 0.3, size=40)
        adv = rng.normal(size=40)
        eps = 0.2
        loss, dlogp, st = ppo_clip_loss(new, old, adv, eps)
        rho = np.exp(new - old)
        self.assertAlmostEqual(loss, -np.mean(np.minimum(rho * adv, np.clip(rho, 1 - eps, 1 + eps) * adv)))
        kink = np.isclose(np.abs(rho - 1), eps, atol=1e-4)
        num = num_grad(lambda: ppo_clip_loss(new, old, adv, eps)[0], new)
        np.testing.assert_allclose(dlogp[~kink], num[~kink], atol=1e-7)
        self.assertAlmostEqual(st["clip_frac"], float((np.abs(rho - 1) > eps).mean()))
        self.assertGreaterEqual(st["approx_kl"], 0.0)

    def test_sgd_is_literal_update(self):
        p = {"w": np.array([1.0, 2.0])}
        SGD(p).step({"w": np.array([0.5, -1.0])}, lr=0.1)
        np.testing.assert_allclose(p["w"], [0.95, 2.1])


if __name__ == "__main__":
    unittest.main()
