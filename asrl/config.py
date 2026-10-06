"""Configuration for Agentic SCALE-RL-PPO.

Every numeric default below is an *implementation choice*: neither the DOCX,
the PPT nor the SCALE-RL paper gives concrete values for alpha, B, eps, b0,
w0..w3, N, gamma, lambda_GAE, K, M, I, h0, K_meta, I_probe or delta.

Parameters are grouped by who may change them:

* ``RewardConfig``  - reward-defining (alpha, B, eps, b0, w0..w3). Fixed for the
  whole run. The meta-agent is NOT allowed to touch these (DOCX: "reward
  parameters excluded, so D_hat and its link to L_SR are unchanged").
* ``Hyperparams``   - h = (beta, c_v, c_q, c_e, eps_clip, eta_theta, eta_psi,
  eta_phi). The only things the meta-agent may change.
* ``PPOConfig``     - structural PPO settings (T, n_env, N, gamma, ...).
* ``MetaConfig``    - outer agentic loop (K_meta, I_probe, delta, H bounds).
"""
from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field, asdict
from typing import Optional


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@dataclass
class DataConfig:
    source: str = "synthetic"          # "synthetic" | "csv"
    csv_path: Optional[str] = None     # wide CSV: date column + one price column per asset
    n_assets: int = 8
    n_days: int = 2520                 # ~10 trading years
    train_frac: float = 0.6
    val_frac: float = 0.2              # test = remainder; strictly chronological
    data_seed: int = 1234              # dataset is fixed across experiment seeds
    lookback: int = 5                  # lagged-return window in the market state s_t
    vol_window: int = 20               # rolling mean/vol features in s_t
    transaction_cost: float = 0.0005   # proportional cost on turnover (5 bp)
    start_date: str = "2012-01-02"


# ----------------------------------------------------------------------------
# Reward-defining parameters (frozen; never touched by the meta-agent)
# ----------------------------------------------------------------------------
@dataclass
class RewardConfig:
    alpha: float = 0.02                # moment adaptation rate -> W = ceil(1/alpha) = 50
    burn_in: Optional[int] = None      # B; None -> W (so the E_cal/Turb windows are full when D_hat starts)
    eps: float = 1e-6                  # variance floor (Assumption 1)
    b0: float = 1e-4                   # initial second moment; must satisfy b0 >= eps
    # Gate lambda_t = sigmoid(w0 + w1*E_cal + w2*Turb + w3*ell).
    # Signs: SCALE-RL paper says lambda -> 1 (trust realized return) when E_cal or
    # turbulence is high, so w1, w2 > 0. w3 > 0 is OUR ASSUMPTION (high event risk ->
    # trust the critic forecast less). Turb is a raw squared Mahalanobis distance whose
    # mean is ~ n_assets, hence the small w2.
    w0: float = -2.0
    w1: float = 2.0
    w2: float = 0.1
    w3: float = 2.0
    # "docx": line 14 literally, (1/W) sum_j (1/N) sum_i |1{r_j<=z_i}-tau_i|  (floor ~1/3 when calibrated)
    # "coverage": (1/N) sum_i | mean_j 1{r_j<=z_i} - tau_i |  (0 when calibrated; matches paper's 0.255->0.005 scale)
    cal_error_mode: str = "docx"
    # turbulence window excludes y_{t-1} (Kritzman-Li convention). Set True to include it.
    turb_window_includes_current: bool = False

    @property
    def W(self) -> int:
        return int(math.ceil(1.0 / self.alpha))

    @property
    def B(self) -> int:
        return self.W if self.burn_in is None else int(self.burn_in)

    def validate(self) -> None:
        assert 0.0 < self.alpha < 1.0
        assert self.eps > 0.0
        assert self.b0 >= self.eps, "DOCX line 8 requires b0 >= eps"
        assert self.B >= 0
        assert self.cal_error_mode in ("docx", "coverage")


REWARD_PARAM_NAMES = ("alpha", "burn_in", "eps", "b0", "w0", "w1", "w2", "w3")


# ----------------------------------------------------------------------------
# Optimisation hyper-parameters h (the meta-agent's search space)
# ----------------------------------------------------------------------------
@dataclass
class Hyperparams:
    beta: float = 1.0          # weight of L_cal relative to L_pin
    c_v: float = 0.5           # value-loss weight
    c_q: float = 1.0           # quantile-loss weight
    c_e: float = 1e-3          # entropy weight
    eps_clip: float = 0.2      # PPO clip range
    eta_theta: float = 3e-4    # actor learning rate
    eta_psi: float = 1e-3      # value-head learning rate
    eta_phi: float = 1e-3      # quantile-head learning rate

    def as_dict(self) -> dict:
        return asdict(self)

    def copy(self) -> "Hyperparams":
        return dataclasses.replace(self)


HYPERPARAM_NAMES = tuple(f.name for f in dataclasses.fields(Hyperparams))


# ----------------------------------------------------------------------------
# PPO structure
# ----------------------------------------------------------------------------
@dataclass
class PPOConfig:
    T: int = 200               # episode length (steps)
    n_env: int = 8             # parallel environments per rollout
    iterations: int = 40       # I
    gamma: float = 0.999       # (1-gamma)*T = 0.2 << 1 as DOCX recommends
    lambda_gae: float = 0.95
    epochs: int = 4            # K
    minibatch: int = 256       # M
    n_quantiles: int = 16      # N
    hidden: int = 64
    n_hidden_layers: int = 2
    # "adam" (default, practical) or "sgd" (DOCX line 34 literally: theta <- theta - eta*grad).
    # NOTE: with separate parameter groups, Adam is nearly invariant to constant loss
    # scaling, so c_v and c_q have little effect under Adam. Use "sgd" for the literal update.
    optimizer: str = "adam"
    log_std_init: float = -0.5
    log_std_bounds: tuple = (-5.0, 1.0)  # numerical guard, not in the DOCX
    quantile_out_scale: float = 0.01     # parameterisation: z = 0.01 * net(x); returns are O(1e-2)
    eval_every: int = 5                  # validation logging cadence (iterations)


# ----------------------------------------------------------------------------
# Meta-agent (outer loop)
# ----------------------------------------------------------------------------
@dataclass
class MetaConfig:
    K_meta: int = 5            # PPO iterations between meta-agent calls
    I_probe: int = 3           # probe iterations from the current checkpoint
    delta: float = 0.05        # acceptance margin on (annualised) validation Sharpe
    agent: str = "rule"        # "rule" (offline mock) | "llm" (needs a backend callable)
    # "checkpoint": S_V(h) = validation Sharpe of the current checkpoint (literal, cheap)
    # "control":    S_V(h) = validation Sharpe after I_probe iterations under h (fairer, 2x cost)
    baseline_mode: str = "checkpoint"
    # Bounded discrete proposal set H: multiplicative steps inside box bounds.
    bounds: dict = field(default_factory=lambda: {
        "beta": (0.1, 100.0),
        "c_v": (0.05, 2.0),
        "c_q": (0.1, 10.0),
        "c_e": (1e-5, 0.05),
        "eps_clip": (0.05, 0.4),
        "eta_theta": (1e-5, 3e-3),
        "eta_psi": (1e-5, 1e-2),
        "eta_phi": (1e-5, 1e-2),
    })
    steps: dict = field(default_factory=lambda: {
        "beta": (0.5, 2.0),
        "c_v": (0.5, 2.0),
        "c_q": (0.5, 2.0),
        "c_e": (0.5, 2.0),
        "eps_clip": (0.75, 1.5),
        "eta_theta": (0.5, 2.0),
        "eta_psi": (0.5, 2.0),
        "eta_phi": (0.5, 2.0),
    })


# ----------------------------------------------------------------------------
# Variant switches (benchmarks and ablations)
# ----------------------------------------------------------------------------
@dataclass
class VariantConfig:
    name: str = "agentic_full"
    reward: str = "gated_dsr"        # "return" (standard PPO) | "gated_dsr"
    use_quantile: bool = True        # quantile head Z_phi + L_pin + L_cal
    use_ab_state: bool = True        # A_{t-1}, B_{t-1} in the state
    use_llm_risk: bool = True        # ell_t in state and in the gate (w3)
    use_llm_embedding: bool = True   # e_t in state
    gate_fixed_one: bool = False     # ablation: lambda_t == 1 (classical DSR moments)
    use_meta: bool = True            # LLM meta-agent outer loop


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    h0: Hyperparams = field(default_factory=Hyperparams)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    signal_lookback_days: int = 10   # text window fed to the (mock) LLM; see README
    embedding_dim: int = 16
    llm_knowledge_cutoff: Optional[str] = None  # set for a real LLM; test must start after it

    def to_dict(self) -> dict:
        return asdict(self)

    def copy(self) -> "ExperimentConfig":
        return copy.deepcopy(self)


def preset(name: str) -> ExperimentConfig:
    """Named presets. 'smoke' is for CI (< ~1 min), 'quick' is the smallest viable
    experiment, 'full' is a longer run."""
    cfg = ExperimentConfig()
    if name == "smoke":
        cfg.data.n_days = 900
        cfg.ppo.iterations = 4
        cfg.ppo.n_env = 2
        cfg.ppo.T = 120
        cfg.ppo.epochs = 2
        cfg.ppo.minibatch = 120
        cfg.ppo.eval_every = 2
        cfg.meta.K_meta = 2
        cfg.meta.I_probe = 1
    elif name == "quick":
        pass  # defaults above
    elif name == "full":
        cfg.ppo.iterations = 150
        cfg.ppo.n_env = 16
        cfg.meta.K_meta = 10
        cfg.meta.I_probe = 5
    else:
        raise ValueError(f"unknown preset {name!r}")
    return cfg
