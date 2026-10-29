"""Variable importance for causal forests (Benard & Josse, 2023), for EconML's CausalForestDML.

Python port of the authors' R function ``vimp_causal_forests`` (built on grf). For each group
of variables J, the final-stage causal forest is retrained without J on the *same* nuisance
residuals, and the corrected retrained prediction of equation (4) is compared with the original
out-of-bag CATEs:

    I_n(J) = sum_i (tau_hat_i - theta_i^(-J))^2 / (n * var(tau_hat))  -  I_n(0)

where I_n(0) is the same score for a forest retrained on all columns (retraining bias).

EconML has no equivalent of grf's ``get_forest_weights``, so the retrained forest's out-of-bag
weights alpha are rebuilt from its subsample and honest-split indices. Products ``alpha @ v``
are computed tree by tree as leaf means, without materialising the n x n matrix.
"""

import numpy as np
import pandas as pd
from econml.grf import CausalForest


def oob_weighted_means(forest, X, V):
    """Compute ``alpha @ V`` for the out-of-bag forest weights of a fitted econml CausalForest.

    alpha[i, k] = mean over trees t with i out-of-bag of
                  1{k in estimation half of t, leaf_t(k) == leaf_t(i)} / |estimation half in leaf_t(i)|
    """
    X = np.asarray(X, dtype=np.float64)
    V = np.column_stack([np.ones(X.shape[0]), np.asarray(V, dtype=np.float64)])
    acc = np.zeros(V.shape)
    for tree, sub in zip(forest.estimators_, forest.get_subsample_inds()):
        _, val_local = tree.get_train_test_split_inds()
        val = sub[val_local]
        leaves = tree.apply(X)
        n_nodes = tree.tree_.node_count
        counts = np.bincount(leaves[val], minlength=n_nodes)
        sums = np.column_stack([np.bincount(leaves[val], weights=V[val, c], minlength=n_nodes)
                                for c in range(V.shape[1])])
        leaf_means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
        oob = np.ones(X.shape[0], dtype=bool)
        oob[sub] = False
        acc[oob] += leaf_means[leaves[oob]]
    return acc[:, 1:] / acc[:, [0]]


def compute_vimp(forest, X, Y_centered, W_centered, tau_hat):
    """Importance score from the retrained forest, as in the R compute_vimp."""
    V = np.column_stack([W_centered, W_centered ** 2, Y_centered, tau_hat,
                         W_centered * Y_centered, tau_hat * W_centered ** 2])
    W_bar, W2_bar, Y_bar, tau_bar, WY_bar, tauW2_bar = oob_weighted_means(forest, X, V).T
    tau_hat_new = ((WY_bar - tauW2_bar) - (W_bar * Y_bar - tau_bar * W2_bar)) / (W2_bar - W_bar ** 2)
    return float(np.sum((tau_hat - tau_hat_new) ** 2) / (len(Y_centered) * np.var(tau_hat, ddof=1)))


def vimp_causal_forest(cf, X, groups=None, seed=20230807):
    """Group variable importance for a CausalForestDML fitted with ``cache_values=True``.

    Parameters
    ----------
    cf : econml.dml.CausalForestDML, fitted with cache_values=True
    X : pandas.DataFrame, the covariate matrix cf was fitted on
    groups : dict of group name -> list of column names. Default None: each column is a group.
    seed : int, seeds the retrained forests

    Returns
    -------
    pandas.Series of importances indexed by group, sorted in decreasing order.
    ``.attrs["baseline"]`` holds the retraining bias I_n(0).
    """
    # get initial data: the cross-fitted residuals of the original fit
    if not hasattr(cf, "residuals_"):
        raise ValueError("cf must be a CausalForestDML fitted with cache_values=True.")
    Y_centered, W_centered, _, _ = cf.residuals_
    Y_centered, W_centered = np.ravel(Y_centered), np.ravel(W_centered)
    X_arr = np.asarray(X, dtype=np.float64)
    columns = list(X.columns)
    p = X_arr.shape[1]

    # get oob predictions of the original final forest
    forest = cf.model_final_.estimators_[0]
    tau_hat = forest.oob_predict(X_arr).ravel()

    # set variable groups
    if groups is None:
        groups = {c: [c] for c in columns}
    if not isinstance(groups, dict) or len(groups) == 0:
        raise ValueError("groups must be a non-empty dict.")
    if not all(c in columns for cols in groups.values() for c in cols):
        raise ValueError("Variable names provided in groups not all found in input data.")
    if max(len(set(cols)) for cols in groups.values()) >= p:
        raise ValueError("A group cannot contain all input variables.")
    index_groups = {name: [columns.index(c) for c in dict.fromkeys(cols)] for name, cols in groups.items()}

    # retrain with the settings of the original final forest, mtry capped at the remaining columns
    params = forest.get_params()
    seeds = [int(s) for s in np.random.SeedSequence(seed).generate_state(len(index_groups) + 1)]

    def retrain_and_score(keep, s):
        p_keep = dict(params, random_state=s)
        if isinstance(p_keep["max_features"], (int, np.integer)) and not isinstance(p_keep["max_features"], bool):
            p_keep["max_features"] = int(min(p_keep["max_features"], len(keep)))
        X_keep = np.ascontiguousarray(X_arr[:, keep])
        c_forest = CausalForest(**p_keep).fit(X_keep, W_centered.reshape(-1, 1), Y_centered.reshape(-1, 1))
        return compute_vimp(c_forest, X_keep, Y_centered, W_centered, tau_hat)

    # compute vimp for all variable groups
    In = pd.Series({name: retrain_and_score([j for j in range(p) if j not in index], s)
                    for (name, index), s in zip(index_groups.items(), seeds[1:])}, name="importance")

    # compute retrain bias
    In0 = retrain_and_score(list(range(p)), seeds[0])

    # compute debiased importance
    In = (In - In0).sort_values(ascending=False)
    In.index.name = "group"
    In.attrs["baseline"] = In0
    return In
