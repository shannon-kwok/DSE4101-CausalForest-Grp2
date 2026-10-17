"""Validation of src/vimp.py against the simulations of Benard & Josse (2023).

Run with ``python -m pytest tests/test_vimp.py`` or ``python tests/test_vimp.py`` (about 11 minutes on 16 cores).
Simulations use the paper's n = 3000 but 1000 trees instead of 2000, and a single repetition.
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.grf import CausalForest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vimp import oob_weighted_means, vimp_causal_forest  # noqa: E402

N = 3000
N_TREES = 1000


def _fit_cf(X, W, Y, seed=0):
    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=seed),
        model_t=RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=seed),
        discrete_treatment=True, n_estimators=N_TREES, cv=5, random_state=seed)
    cf.fit(Y, W, X=X.to_numpy(), cache_values=True)
    return cf


def _experiment1(oracle_centering=False):
    """p=8, X ~ N(0, Sigma) with Cov(X1, X5)=0.9; tau = X1 1{X1>0} + 0.6 X2 1{X2>0}; X1 also confounds."""
    rng = np.random.default_rng(1)
    cov = np.eye(8)
    cov[0, 4] = cov[4, 0] = 0.9
    X = rng.multivariate_normal(np.zeros(8), cov, size=N)
    W = rng.binomial(1, 0.4 + 0.2 * (X[:, 0] > 0))
    tau = X[:, 0] * (X[:, 0] > 0) + 0.6 * X[:, 1] * (X[:, 1] > 0)
    mu = (X[:, 2] * X[:, 3]) ** 2
    Y = tau * W + mu + rng.normal(0, 0.1, N)
    if oracle_centering:
        Y = Y - mu  # removes the hard-to-centre baseline that leaks into X3/X4
    return pd.DataFrame(X, columns=[f"X{j}" for j in range(1, 9)]), W, Y


@lru_cache(maxsize=None)
def exp1_model(oracle_centering=False):
    X, W, Y = _experiment1(oracle_centering)
    return _fit_cf(X, W, Y), X


@lru_cache(maxsize=None)
def exp1_vimp(seed=20230807, oracle_centering=False):
    cf, X = exp1_model(oracle_centering)
    return vimp_causal_forest(cf, X, {c: [c] for c in X.columns}, seed=seed, verbose=False)


@lru_cache(maxsize=None)
def exp3_vimp():
    """p=5 uniform, X1 = U^3, pi(X) = X1, tau = 10 X1 (1 - X1): the corrective term matters."""
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(N, 5))
    X[:, 0] = rng.uniform(size=N) ** 3
    W = rng.binomial(1, X[:, 0])
    Y = 10 * X[:, 0] * (1 - X[:, 0]) * W + X[:, 1] + rng.normal(0, 0.1, N)
    X = pd.DataFrame(X, columns=[f"X{j}" for j in range(1, 6)])
    cf = _fit_cf(X, W, Y)
    return vimp_causal_forest(cf, X, {c: [c] for c in X.columns}, also_uncorrected=True, verbose=False)


# --------------------------------------------------------------------------------------------
# Forest weights
# --------------------------------------------------------------------------------------------

def test_oob_weights_reproduce_econml_oob_predict():
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(size=(n, 4))
    W = rng.integers(0, 2, n) - 0.5
    Y = X[:, 0] * W + rng.normal(size=n)
    for kw in [{}, {"honest": False}, {"inference": False, "max_samples": 0.7}, {"min_samples_leaf": 20}]:
        forest = CausalForest(n_estimators=100, random_state=3, **kw).fit(X, W.reshape(-1, 1), Y.reshape(-1, 1))
        W_bar, W2_bar, Y_bar, WY_bar = oob_weighted_means(forest, X, np.column_stack([W, W ** 2, Y, W * Y])).T
        rebuilt = (WY_bar - W_bar * Y_bar) / (W2_bar - W_bar ** 2)
        assert np.max(np.abs(rebuilt - forest.oob_predict(X).ravel())) < 1e-10, kw


# --------------------------------------------------------------------------------------------
# Paper simulations
# --------------------------------------------------------------------------------------------

def test_experiment1_ranking():
    imp = exp1_vimp()["importance"]
    assert set(imp.index[:2]) == {"X1", "X2"}, imp
    assert imp["X1"] > 0.1 and imp["X2"] > 0.1, imp
    assert abs(imp["X5"]) < 0.03, imp  # correlated with X1 but not a heterogeneity driver
    assert (imp[["X6", "X7", "X8"]].abs() < 0.03).all(), imp


def test_experiment1_oracle_centering_matches_theoretical_importance():
    # With the baseline mu removed exactly, the only remaining error is forest estimation error,
    # so values should be close to the oracle I = 0.26 (X2) and 0.18 (X1), and 0 elsewhere.
    imp = exp1_vimp(oracle_centering=True)["importance"]
    assert abs(imp["X2"] - 0.26) < 0.06, imp
    assert abs(imp["X1"] - 0.18) < 0.06, imp
    assert (imp.drop(["X1", "X2"]).abs() < 0.02).all(), imp


def test_experiment3_correction_removes_bias():
    res = exp3_vimp()
    corrected, uncorrected = res["importance"], res["importance_uncorrected"]
    assert abs(corrected["X1"] - 1.0) < 0.15, res  # paper: 0.98
    assert uncorrected["X1"] > 1.4, res  # paper: 1.57; Theorem 5 bias is about 0.72
    assert uncorrected["X1"] - corrected["X1"] > 0.4, res
    assert (corrected.drop("X1").abs() < 0.03).all(), res


def test_noise_and_overlapping_groups():
    cf, X = exp1_model()
    groups = {"signal": ["X1", "X2"], "noise": ["X6", "X7", "X8"], "overlap": ["X2", "X5"]}
    imp = vimp_causal_forest(cf, X, groups, verbose=False)["importance"]
    assert abs(imp["noise"]) < 0.03, imp
    assert imp["signal"] > 0.4, imp
    assert imp["overlap"] > 0.1, imp


def test_stable_across_seeds():
    a, b = exp1_vimp(seed=1)["importance"], exp1_vimp(seed=2)["importance"]
    assert set(a.index[:2]) == set(b.index[:2]) == {"X1", "X2"}
    assert (a - b.reindex(a.index)).abs().max() < 0.05, pd.concat([a, b], axis=1)


# --------------------------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------------------------

def test_input_validation_and_drill_down():
    cf, X = exp1_model()
    for bad in [{}, {"g": []}, {"g": ["not_a_column"]}, {"g": list(X.columns)}]:
        try:
            vimp_causal_forest(cf, X, bad, verbose=False)
        except ValueError:
            continue
        raise AssertionError(f"groups={bad} should have been rejected")

    res = vimp_causal_forest(cf, X, {"signal": ["X1", "X2"], "rest": ["X3"]}, drill_down="signal", verbose=False)
    assert set(res.index) == {"X1", "X2"}

    no_cache = CausalForestDML(discrete_treatment=True, n_estimators=8, cv=2)
    no_cache.fit(np.zeros(100) + np.arange(100) % 2, np.arange(100) % 2, X=X.to_numpy()[:100])
    try:
        vimp_causal_forest(no_cache, X.iloc[:100], {"g": ["X1"]}, verbose=False)
    except ValueError:
        pass
    else:
        raise AssertionError("a model fitted without cache_values=True should be rejected")


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
