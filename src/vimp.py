"""Variable importance for causal forests (Benard & Josse, 2023), for EconML's CausalForestDML.

Python port of the authors' R function ``vimp_causal_forests`` (built on grf). For each group
of variables J, the final-stage causal forest is retrained without J on the *same* nuisance
residuals (centering always uses all variables, Proposition 4), and the corrected retrained
prediction of equation (4) is compared with the original out-of-bag CATEs:

    I_n(J) = sum_i (tau_hat_i - theta_i^(-J))^2 / (n * var(tau_hat))  -  I_n(0)

where I_n(0) is the same score for a forest retrained on all columns with a different seed
(retraining noise). Small negative values mean "no importance".

The retrained forest's out-of-bag weights alpha are rebuilt exactly from EconML's own
subsample and honest-split indices, and never materialised as an n x n matrix: every product
``alpha @ v`` is computed tree by tree as leaf means over the honest estimation half,
averaged over the trees for which the row is out-of-bag. For every refit, the uncorrected
formula with these weights is checked against EconML's ``oob_predict``.
"""

import time

import numpy as np
import pandas as pd
from econml.grf import CausalForest
from joblib import Parallel, delayed


# --------------------------------------------------------------------------------------------
# Out-of-bag forest weights
# --------------------------------------------------------------------------------------------

def oob_weighted_means(forest, X, V):
    """Compute ``alpha @ V`` for the out-of-bag forest weights of a fitted econml CausalForest.

    alpha[i, k] = mean over trees t with i out-of-bag of
                  1{k in estimation half of t, leaf_t(k) == leaf_t(i)} / |estimation half in leaf_t(i)|

    Rows are normalised to sum to one, which is what grf's ``get_forest_weights`` returns
    and what EconML's ``oob_predict`` implies (its moment solve is invariant to that scaling).

    Parameters
    ----------
    forest : econml.grf.CausalForest, fitted on X without sample weights and without warm_start
    X : array of shape (n, p), exactly the matrix the forest was fitted on
    V : array of shape (n, k)

    Returns
    -------
    means : array of shape (n, k), NaN for rows that are in-bag for every tree
    """
    X = np.asarray(X, dtype=np.float64)
    V = np.column_stack([np.ones(X.shape[0]), np.asarray(V, dtype=np.float64)])
    n, k = V.shape
    acc = np.zeros((n, k))
    for tree, sub in zip(forest.estimators_, forest.get_subsample_inds()):
        _, val_local = tree.get_train_test_split_inds()
        val = sub[val_local]
        leaves = tree.apply(X)
        n_nodes = tree.tree_.node_count
        val_leaves = leaves[val]
        counts = np.bincount(val_leaves, minlength=n_nodes)
        sums = np.column_stack([np.bincount(val_leaves, weights=V[val, c], minlength=n_nodes)
                                for c in range(k)])
        leaf_means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
        oob = np.ones(n, dtype=bool)
        oob[sub] = False
        acc[oob] += leaf_means[leaves[oob]]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = acc[:, 1:] / acc[:, [0]]
    means[acc[:, 0] == 0] = np.nan
    return means


def _retrained_predictions(forest, X_sub, W_c, Y_c, tau_hat):
    """Corrected (equation 4) and uncorrected retrained OOB predictions, as in the R compute_vimp."""
    V = np.column_stack([W_c, W_c ** 2, Y_c, tau_hat, W_c * Y_c, tau_hat * W_c ** 2])
    W_bar, W2_bar, Y_bar, tau_bar, WY_bar, tauW2_bar = oob_weighted_means(forest, X_sub, V).T
    denom = W2_bar - W_bar ** 2
    tau_uncorrected = (WY_bar - W_bar * Y_bar) / denom
    tau_corrected = ((WY_bar - tauW2_bar) - (W_bar * Y_bar - tau_bar * W2_bar)) / denom
    return tau_corrected, tau_uncorrected


def _score(tau_hat, tau_new):
    """sum((tau_hat - tau_new)^2) / (n * var(tau_hat)), over rows where both are defined."""
    ok = np.isfinite(tau_hat) & np.isfinite(tau_new)
    t, s = tau_hat[ok], tau_new[ok]
    return float(np.sum((t - s) ** 2) / (len(t) * np.var(t, ddof=1)))


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------

def _validate_groups(groups, columns):
    """Check groups like the R reference and return {name: [column indices]}."""
    if not isinstance(groups, dict) or len(groups) == 0:
        raise ValueError("groups must be a non-empty dict of group name -> list of column names.")
    columns = list(columns)
    index_groups = {}
    for name, cols in groups.items():
        cols = list(dict.fromkeys([cols] if isinstance(cols, str) else cols))
        if len(cols) == 0:
            raise ValueError(f"Group {name!r} is empty.")
        missing = [c for c in cols if c not in columns]
        if missing:
            raise ValueError(f"Group {name!r} has columns not found in X: {missing}")
        if len(cols) >= len(columns):
            raise ValueError(f"Group {name!r} contains all input variables.")
        index_groups[name] = [columns.index(c) for c in cols]
    return index_groups


def _extract_inputs(cf, X):
    """Pull residuals, the original final forest and its OOB CATEs from a fitted CausalForestDML."""
    try:
        Y_res, T_res, X_cached, _ = cf.residuals_
    except AttributeError as e:
        raise ValueError("cf must be a CausalForestDML fitted with cache_values=True, so that its "
                         "nuisance residuals can be reused.") from e
    if getattr(cf, "featurizer", None) is not None:
        raise ValueError("CausalForestDML with a featurizer is not supported.")
    X_arr = np.asarray(X, dtype=np.float64)
    if X_cached is None or not np.array_equal(np.asarray(X_cached, dtype=np.float64), X_arr):
        raise ValueError("X must be exactly the covariate matrix (same rows, same order) cf was fitted on.")
    T_res = np.asarray(T_res, dtype=np.float64)
    Y_res = np.asarray(Y_res, dtype=np.float64)
    if (T_res.ndim == 2 and T_res.shape[1] != 1) or (Y_res.ndim == 2 and Y_res.shape[1] != 1):
        raise ValueError("Only a single binary treatment and a single outcome are supported.")
    forest = cf.model_final_.estimators_[0]
    if getattr(forest, "warm_start_", False):
        raise ValueError("Forests fitted with warm_start=True are not supported.")
    tau_hat = forest.oob_predict(X_arr).ravel()
    return X_arr, Y_res.ravel(), T_res.ravel(), forest, tau_hat


def _cap_max_features(params, n_columns):
    """Cap an integer max_features (grf's mtry) at the remaining column count, as in the R reference."""
    params = dict(params)
    mf = params["max_features"]
    if isinstance(mf, (int, np.integer)) and not isinstance(mf, bool):
        params["max_features"] = int(min(mf, n_columns))
    return params


# --------------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------------

def _refit_and_score(keep, X, Y_c, W_c, tau_hat, params, seed, check_weights, tol):
    """Retrain the final forest on X[:, keep] with a new seed and score it against tau_hat."""
    start = time.perf_counter()
    X_sub = np.ascontiguousarray(X[:, keep])
    p = dict(_cap_max_features(params, len(keep)), random_state=int(seed))
    forest = CausalForest(**p).fit(X_sub, W_c.reshape(-1, 1), Y_c.reshape(-1, 1))
    tau_corr, tau_unc = _retrained_predictions(forest, X_sub, W_c, Y_c, tau_hat)
    max_dev = np.nan
    if check_weights:
        ref = forest.oob_predict(X_sub).ravel()
        ok = np.isfinite(ref) & np.isfinite(tau_unc)
        max_dev = float(np.max(np.abs(ref[ok] - tau_unc[ok])))
        if max_dev > tol:
            raise RuntimeError(f"Rebuilt OOB weights do not reproduce econml's oob_predict "
                               f"(max abs deviation {max_dev:.2e} > {tol:.0e}); scores would not be trustworthy.")
    return {"raw": _score(tau_hat, tau_corr), "raw_uncorrected": _score(tau_hat, tau_unc),
            "weight_check_max_dev": max_dev, "fit_seconds": time.perf_counter() - start}


def vimp_causal_forest(cf, X, groups, also_uncorrected=False, drill_down=None, seed=20230807,
                       n_jobs=1, forest_n_jobs=-1, check_weights=True, tol=1e-6, verbose=True):
    """Group variable importance for a fitted CausalForestDML (Benard & Josse, 2023).

    Parameters
    ----------
    cf : econml.dml.CausalForestDML
        Fitted with ``cache_values=True`` on a single binary treatment and single outcome, without
        a featurizer. Its cross-fitted nuisance residuals and its final forest's hyperparameters are
        reused for every retrained forest; the nuisance models are never refitted.
    X : pandas.DataFrame
        Exactly the covariate matrix cf was fitted on, with column names.
    groups : dict of str -> list of str
        Group name -> column names. Each group must be non-empty and must not contain all columns;
        groups may overlap.
    also_uncorrected : bool, default False
        Also return the score without the corrective term of equation (4). Its asymptotic upward bias
        is given by Theorem 5 and vanishes when the propensity is constant.
    drill_down : str or None, default None
        Name of one group in ``groups``. If given, every column of that group is scored as its own
        group, instead of scoring ``groups``.
    seed : int
        Master seed for the retrained forests. The baseline and each group get distinct derived seeds,
        none equal to the original forest's random_state.
    n_jobs : int, default 1
        Number of groups refitted in parallel (joblib processes).
    forest_n_jobs : int or None, default -1
        Threads used inside each forest fit. Does not change results. None keeps the original setting.
    check_weights : bool, default True
        For every refit, verify that the rebuilt OOB weights reproduce econml's oob_predict.
    tol : float
        Tolerance for that check.

    Returns
    -------
    pandas.DataFrame indexed by group, sorted by importance, with columns
        importance                  corrected, debiased score  I_n(J) = raw - I_n(0)
        importance_uncorrected      (if also_uncorrected) same without the corrective term
        raw, raw_uncorrected        scores before subtracting the baseline
        n_columns, columns          group contents
        fit_seconds                 time to refit and score this group
    ``df.attrs`` holds the baseline scores, seeds and total runtime.
    """
    start = time.perf_counter()
    X_arr, Y_c, W_c, forest, tau_hat = _extract_inputs(cf, X)
    columns = list(X.columns) if hasattr(X, "columns") else [f"X{j}" for j in range(X_arr.shape[1])]

    if drill_down is not None:
        if drill_down not in groups:
            raise ValueError(f"drill_down={drill_down!r} is not a group name.")
        cols = groups[drill_down]
        groups = {c: [c] for c in ([cols] if isinstance(cols, str) else cols)}
    index_groups = _validate_groups(groups, columns)

    params = forest.get_params()
    if forest_n_jobs is not None:
        # Does not change the fitted forest: econml draws all subsamples and tree seeds before
        # its (threaded) parallel loop.
        params["n_jobs"] = forest_n_jobs
    seeds = [int(s) for s in np.random.SeedSequence(seed).generate_state(len(index_groups) + 1)]
    seeds = [s + 1 if s == params.get("random_state") else s for s in seeds]

    all_cols = list(range(X_arr.shape[1]))
    jobs = [("__baseline__", all_cols, seeds[0])]
    jobs += [(name, [j for j in all_cols if j not in idx], s)
             for (name, idx), s in zip(index_groups.items(), seeds[1:])]
    if verbose:
        print(f"Refitting {len(jobs)} forests ({len(jobs) - 1} groups + 1 baseline) on n={X_arr.shape[0]:,}...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_refit_and_score)(keep, X_arr, Y_c, W_c, tau_hat, params, s, check_weights, tol)
        for _, keep, s in jobs)

    base = results[0]
    rows = []
    for (name, _, _), res in zip(jobs[1:], results[1:]):
        cols = [columns[j] for j in index_groups[name]]
        row = {"importance": res["raw"] - base["raw"]}
        if also_uncorrected:
            row["importance_uncorrected"] = res["raw_uncorrected"] - base["raw_uncorrected"]
        row.update(raw=res["raw"])
        if also_uncorrected:
            row["raw_uncorrected"] = res["raw_uncorrected"]
        row.update(n_columns=len(cols), columns=", ".join(cols), fit_seconds=round(res["fit_seconds"], 1))
        rows.append(pd.Series(row, name=name))
    out = pd.DataFrame(rows).sort_values("importance", ascending=False)
    out.index.name = "group"
    out.attrs.update(baseline=base["raw"], baseline_uncorrected=base["raw_uncorrected"],
                     baseline_seed=seeds[0], group_seeds=dict(zip(index_groups, seeds[1:])),
                     max_weight_check_dev=max(r["weight_check_max_dev"] for r in results),
                     total_seconds=time.perf_counter() - start)
    if verbose:
        print(f"Done in {out.attrs['total_seconds'] / 60:.1f} min. Baseline I_n(0) = {base['raw']:.4f}")
    return out
