# Project Summary: DSE4101 Causal Forest (Group 2)

## 1. What the project is about

This is the DSE4101 (Digital Economy) capstone project for Group 2. It asks whether an **SMS marketing campaign** changed customers' **purchase behaviour**, and **which customers it affected more or less**.

It uses the **X5 RetailHero uplift dataset**, which comes from a randomised marketing experiment run by X5 Retail Group, a Russian grocery retailer:

- **Treatment (`D` / `Z`)**: whether the customer received the SMS communication. Assignment was randomised, about 50/50.
- **Outcome (`Y`, `target`)**: whether the customer made a purchase after the campaign (binary, about 62% = 1).
- **Covariates (`X`)**: customer characteristics from before the campaign, built from demographics and purchase history.

The analysis moves from simple to more flexible methods:

1. **Average Treatment Effect (ATE)**: does the SMS work on average?
2. **Pre-specified and LASSO-selected logistic regressions with treatment interactions**: is there heterogeneity along a few named or data-selected dimensions?
3. **Causal Forest (`econml.CausalForestDML`)**: estimate an individual-level **Conditional Average Treatment Effect (CATE)** for every customer, without fixing the functional form in advance.

The ATE benchmarks also act as a sanity check: the mean of the causal forest's CATEs should come out close to the experimental ATE.

---

## 2. Repository structure

```
DSE4101-CausalForest-Grp2/
├── README.md                     # Setup instructions
├── requirements.txt              # Pinned environment (pip freeze, UTF-16 encoded)
├── .gitignore
├── src/
│   └── data.py                   # Downloads the X5 dataset via scikit-uplift
├── scripts/                      # Currently empty (only __pycache__)
├── data/
│   ├── raw/x5/                   # clients.csv.gz, uplift_train.csv.gz (tracked)
│   │                             # purchases.csv.gz (large, downloaded, not tracked)
│   ├── Processed/
│   │   └── retailhero_causal_forest_features.csv   # Modelling table (200,039 × 25)
│   └── output/
│       ├── lasso_selected_features.csv             # 5 features chosen by LASSO
│       └── cf_cate_results.csv                     # Test-set CATE per customer (60,012 rows)
└── notebooks/
    ├── EDA.ipynb                 # Loads raw files, inspects headers
    ├── retailhero.ipynb          # Feature engineering → processed CSV
    ├── ate_benchmark.ipynb       # ATE: difference-in-means + density-weighted (X5 paper)
    ├── lasso.ipynb               # LASSO selection of treatment interactions
    ├── logistic_regression.ipynb # Pre-specified + LASSO-informed logit models
    └── causal_forest.ipynb       # Causal forest CATE estimation
```

### Key dependencies (from `requirements.txt`)
`pandas 2.3.3`, `numpy 2.2.6`, `scikit-learn 1.7.2`, `scikit-uplift 0.5.1` (dataset loader), `econml 0.17.0` (causal forest), `statsmodels 0.15.0` (logit), `scipy 1.15.3`, `matplotlib 3.10.9`, `joblib 1.6.0`.

### Setup
```bash
pip install -r requirements.txt
python -m src.data      # downloads purchases.csv.gz (~45.8M rows) into data/raw/x5/
```

---

## 3. Pipeline, notebook by notebook

Suggested run order: `src/data.py` → `EDA` → `retailhero` → `ate_benchmark` → `lasso` → `logistic_regression` → `causal_forest`.

### 3.1 `src/data.py`: data download
Calls `sklift.datasets.fetch_x5()` to download the three raw X5 tables into `data/raw/x5/`:

| Table | Shape | Content |
|---|---|---|
| `clients` | 400,162 × 5 | Demographics (age, gender, dates) |
| `purchases` | 45,786,568 × 13 | Line-item transaction history before the campaign |
| `uplift_train` | 200,039 rows | `client_id`, `treatment_flg`, `target` |

### 3.2 `EDA.ipynb`: initial look
Loads the three raw files and prints their column names and first few rows. Exploration is light.

### 3.3 `retailhero.ipynb`: feature engineering
Builds one row per customer in the training set (200,039 customers), aggregating the purchases table into **22 pre-treatment covariates**:

| Group | Features |
|---|---|
| Demographics | `age` (values outside 0 to 100 replaced with the median, 45), `gender_F`, `gender_M`, `gender_U` (one-hot) |
| Purchase volume | `num_transactions`, `num_unique_products`, `num_stores_visited`, `total_quantity`, `avg_items_per_transaction` |
| Spending | `total_spend`, `avg_transaction_spend`, `median_transaction_spend`, `spend_std` (computed per transaction after de-duplicating line items) |
| Loyalty points | `regular_points_received/spent`, `express_points_received/spent`, `net_regular_points_change`, `net_express_points_change` (points spent are stored as negative values) |
| Recency / frequency | `days_since_last_purchase` (reference date 2019-03-18), `avg_days_between_transactions`, `customer_activity_span` |

It then adds `treatment` and `target`. For customers with a single transaction, `spend_std` and `avg_days_between_transactions` are set to 0. The result is saved as `retailhero_causal_forest_features.csv` (200,039 × 25).

### 3.4 `ate_benchmark.ipynb`: average treatment effect
**Benchmark 1: difference in means** (valid because treatment was randomised)

| | Value |
|---|---|
| n treated / control | 99,981 / 100,058 |
| P(purchase \| treated) | 0.6365 |
| P(purchase \| control) | 0.6033 |
| **ATE** | **0.0332** (SE 0.0022, Welch t = 15.3, p ≈ 6e-53) |
| 95% CI | [0.0290, 0.0375] |

The SMS raises purchase probability by about **3.3 percentage points** on average.

**Benchmark 2: density-weighted ATE** (replicating Eq. 16 of the X5 paper)
1. Apply a signed log1p transform and standardise the 22 covariates, then reduce them to 2D with PCA.
2. Split the 2D space into a 20×20 grid. Bins that lack at least one treated and one control customer are dropped (30 bins, 42 rows, 0.02%).
3. ATE = Σ (bin share × within-bin uplift).

Result: **ATE = 0.0320**. Bootstrap with 500 resamples gives SE 0.0020 and 95% CI [0.0278, 0.0358]. The estimate is robust to grid size: 10×10 gives 0.0321 and 30×30 gives 0.0319.

### 3.5 `lasso.ipynb`: selecting heterogeneity drivers
- Design matrix: `[D, X_std, D × X_std]` with 43 columns. Continuous variables are standardised and `gender_U` is the reference category.
- L1-penalised logistic regression (`saga`) with 5-fold stratified CV on log-loss, using the **1-SE rule** to pick the regularisation strength (C = 0.00336 versus a CV-optimal C of 0.0695).
- **5 of 21 treatment interactions were selected**, saved to `data/output/lasso_selected_features.csv`:

| Feature | Interaction coef. (standardised) |
|---|---|
| `express_points_spent` | −0.057 |
| `age` | +0.040 |
| `gender_F` | +0.032 |
| `num_stores_visited` | −0.014 |
| `spend_std` | −0.011 |

### 3.6 `logistic_regression.ipynb`: interaction models
**Pre-specified models** of the form `Y ~ D + X_k + D·X_k` (statsmodels Logit, raw scale):

| Model | Interaction coef. | p-value | Conclusion |
|---|---|---|---|
| 1. Total spend | −6.6e-06 | < 0.001 | Significant heterogeneity |
| 2. Number of transactions | −0.0018 | 0.081 | Not significant at 5% |
| 3. Days since last purchase | +0.0038 | 0.033 | Significant at 5% |

(A fourth pre-specified model was removed in commit `8e09ad1`.)

**LASSO-informed model**: `D` plus the 5 selected main effects plus their 5 interactions:

| Interaction | Coef. | p-value |
|---|---|---|
| `D × express_points_spent` | −0.0049 | < 0.001 |
| `D × age` | +0.0035 | < 0.001 |
| `D × gender_F` | +0.0455 | 0.020 |
| `D × spend_std` | −8.6e-05 | 0.015 |
| `D × num_stores_visited` | −0.0091 | 0.092 (not significant) |

Four of the five interactions are significant at 5%. Read literally, older customers and female customers respond more to the SMS. Customers who have spent more express points (more negative values) and customers with more variable spending respond less, though the sign on `express_points_spent` is easy to misread because points spent are stored as negatives.

### 3.7 `causal_forest.ipynb`: heterogeneous treatment effects
- **Covariates:** all 21 features (with `gender_U` dropped).
- **Split:** 70/30 train/test, stratified by treatment × outcome, giving 140,027 training and 60,012 test customers.
- **Nuisance models:**
  - Outcome model E[Y|X]: `RandomForestRegressor(n_estimators=100, min_samples_leaf=20)`.
  - Treatment model E[D|X]: `DummyClassifier(strategy="prior")`. The constant propensity is correct because treatment was randomised.
- **Forest:** `CausalForestDML(n_estimators=200, min_samples_leaf=20, cv=3, honest=True, inference=True, discrete_treatment=True)`, with `random_state = 1213`.
- **Output:** per-customer test-set CATEs saved to `data/output/cf_cate_results.csv`. The fitted model is saved as `causal_forest_model.joblib`, which is not tracked in git.

**Results on the test set**

| Statistic | Value |
|---|---|
| Mean CATE | 0.0316 |
| Median CATE | 0.0276 |
| Std. dev. | 0.0292 |
| Range | [−0.045, 0.408] |
| Share with CATE > 0 | 92.6% |
| Share with CATE < 0 | 7.4% |
| Experimental test-set ATE | 0.0332 (difference from mean CATE: −0.0016) |

The mean CATE closely matches the experimental ATE, which is a good calibration check. Most customers have a small positive effect, a minority have estimated negative effects, and the distribution has a long right tail of highly responsive customers.

---

## 4. Main findings so far

1. **The SMS campaign works on average.** It raises purchase probability by about 3.2 to 3.3 percentage points. The effect is highly significant and consistent across estimators (difference in means, density-weighted, causal forest mean CATE).
2. **The effect differs across customers.** The logit interaction models and LASSO selection point to age, gender, express-points usage and spending variability as drivers. The causal forest gives individual CATEs: about 93% of customers are positive and about 7% are possibly negative.
3. **Recency and total spend** also show significant interactions in the simple pre-specified models. Transaction count does not.

---

## 5. Gaps, caveats and possible next steps

**Analysis not done yet**
- No CATE inference or validation yet: no confidence intervals from `cf.effect_interval`, no omnibus heterogeneity test (such as a BLP or calibration test), and no feature importance or SHAP for the forest.
- No targeting evaluation yet, such as Qini or uplift curves, or comparing "treat the top-k% by CATE" against "treat everyone".
- There is no group-level (GATE) breakdown of CATEs by the LASSO-selected features to compare against the logit results.

**Methodological caveats**
- The LASSO selection and the follow-up logit were run on the **same full sample**, so the logit p-values are post-selection and likely too optimistic. A sample split or cross-fitting would fix this.

**Housekeeping issues in the repo**
- **Folder name case:** the folder is `data/Processed` (capital P), but most notebooks use `data/processed`. This works on Windows but will break on macOS or Linux. `ate_benchmark` uses `Processed`.
- `retailhero.ipynb` writes the CSV into the notebook's own working directory rather than `data/Processed/`, so it has to be moved by hand.
- The `features.columns` printout in `retailhero.ipynb` is stale (it shows 20 columns, from before gender and labels were added).
- `src/__pycache__/*.pyc` is committed even though `.gitignore` excludes it, and `scripts/` is empty.
- `requirements.txt` is UTF-16 encoded (from a PowerShell `pip freeze`). Some tools expect UTF-8.
- The `.gitignore` exception for `data/processed/retailhero.ipynb` points to a file that does not exist.
