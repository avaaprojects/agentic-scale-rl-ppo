# Agentic SCALE-RL-PPO — research prototype (numpy, Python only)

Implements the DOCX Algorithm 1 (lines are cited in code comments). Status: 45 unit tests pass; smoke pipeline runs end to end. The quick/full presets have NOT yet been run to completion, so there are no benchmark numbers yet.

## Run
    python -m unittest discover -s tests -t .      # 45 tests (~10 s)
    python run_pipeline.py --preset smoke           # ~10 s sanity run
    python run_pipeline.py --preset quick           # 7 variants x 3 seeds (smallest viable experiment)
Outputs go to results/<preset>_<time>/: config.json, dataset.json, llm_signals.json, summary.md/csv/png,
runs/<variant>/seed_k/{metrics.jsonl, meta_decisions.jsonl, checkpoint_final.npz, result.json}.

## Layout
asrl/data (synthetic + CSV loader, chronological split) · env · models (numpy MLP/actor/value/quantile, Adam/SGD) ·
scale_rl (moments, gated DSR, E_cal, turbulence LW, gate, pinball + L_cal) · llm (signal provider + cache, meta-agent) ·
ppo (GAE, clip loss, trainer + meta loop) · benchmarks (variants, metrics) · experiments (pipeline, report) · tests

## Approximations (NOT an exact reproduction)
- numpy with hand-written gradients instead of PyTorch (finite-difference checked).
- Synthetic market + synthetic news corpus (no data download available). `--data-csv` loads real prices.
- Mock LLM: deterministic keyword/hash provider; real LLM plugs in via asrl/llm/providers.py. Text window = 10 days before t.
- Rule-based meta-agent stands in for the LLM; LLMMetaAgent accepts any llm_fn.
- Adam by default; DOCX line 34 is plain gradient descent (`--optimizer sgd`). Under Adam, c_v and c_q barely matter.
- All hyperparameter values are my choices (none are given in the documents): asrl/config.py.
- Guards not in the DOCX: log-std clamp, pseudo-inverse fallback for a singular LW covariance.

## Ambiguities (configurable, please confirm with professor)
1. E_cal line 14 literal has a ~1/3 floor even when calibrated; `--cal-error-mode coverage` is the vanishing version. Both are reported.
2. Signs: w1, w2 > 0 follow the paper; w3 > 0 is my assumption.
3. Learning-rate ordering enforced: eta_theta <= min(eta_psi, eta_phi).
4. Accept test baseline S_V(h): current checkpoint (default) or `--meta-baseline control`.
5. Separate actor/value/quantile networks; critic input (s_t, a_t); V(s_T)=0; turbulence window excludes y_{t-1}.
6. The first paper (MoSIB) contributes only the propose→probe→accept/rollback idea; branches and routing are not implemented.
