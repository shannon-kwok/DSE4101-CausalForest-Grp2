# Causal Forest Applications in Marketing Campaign Evaluation and Customer Targeting

**DSE4101 Capstone Project, Digital Economy,  Group 2**

## 1. Research Question

This project aims to investigate heterogeneous treatment effects (HTE) of SMS marketing on customer purchasing behaviour using the RetailHero/X5 dataset.

We address two main research questions:
1. Does marketing intervention have heterogeneous causal effects on customer purchasing behavior, and which pre-treatment customer characteristics predict that heterogeneity?
2. Under a hypothetical constraint allowing the firm to contact at most 30% of customers, how does a causal forest-based targeting policy perform compared to simpler benchmarks such as random targeting, top 30% by prior spending, and linear-interaction model based targeting?

## 2. Methodology

The analysis uses customer-level data from the RetailHero/X5 Retail Group SMS marketing experiment, comprising **200,039 experimental customers**.

Our empirical framework consists of the following stages:

| Stage | Method | Purpose |
|---|---|---|
| Data Preparation | Transaction aggregation and feature engineering | Construct customer-level pre-treatment characteristics |
| ATE Estimation | Difference in means and density-weighted estimator | Estimate the average effect of SMS marketing |
| Regression-Based HTE | Linear probability models (LPM) | Examine treatment-effect heterogeneity across pre-specified customer characteristics |
| Exploratory Screening | LASSO logistic regression | Identify candidate treatment interactions |
| Causal Forest | EconML `CausalForestDML` | Estimate conditional average treatment effects (CATEs) |
| Model Validation | Best Linear Predictor (BLP) calibration | Assess the reliability of estimated heterogeneity |
| Variable Importance | Grouped drop-and-relearn approach | Identify customer characteristics informative about treatment-effect heterogeneity |
| Targeting Evaluation | AIPW, Qini curves, AUQ and bootstrap | Compare targeting strategies under a 30% contact constraint |

Targeting strategies include causal forest, linear interaction models, random targeting, and top/bottom historical spending. Treating everyone serves as an unconstrained benchmark.

## 3. Project Structure

```text
DSE4101-CausalForest-Grp2/
├── data/                  # Raw and processed datasets
├── notebooks/             # Data preparation and analysis notebooks
├── src/                   # Reusable functions and utilities
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 4. Local Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/shannon-kwok/DSE4101-CausalForest-Grp2.git
cd DSE4101-CausalForest-Grp2
```

### Step 2: Create a virtual environment (recommended)


```bash
python -m venv .venv
```

### Step 3: Activate the virtual environment

   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

### Step 4: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Launch Jupyter Notebook

```bash
jupyter notebook
```

## 5. Running the Analysis

Run the following command in the VS Code terminal from the project root to fetch the RetailHero/X5 dataset:

```bash
python src/data.py
```

Then, open Jupyter Notebook and execute the notebooks in the following order:

1. `notebooks/retailhero.ipynb` — Data cleaning and feature engineering
2. `notebooks/ate_benchmark.ipynb` — ATE estimation
3. `notebooks/lasso.ipynb` — Exploratory LASSO screening
4. `notebooks/logistic_regression.ipynb` — Regression-based HTE estimation
5. `notebooks/causal_forest.ipynb` — Causal forest training and CATE estimation

After fitting the causal forest, run the variable-importance and targeting-evaluation notebooks for the remaining analyses.

## 6. Contributors

**DSE4101 — Digital Economy, Group 2**

Edena Ong Hui Xin A0283319Y

Ong Hai Liang A0271757U

Shannon Kwok En Yi A0281617B

Tan Sze Ping A0286558J

Wang Ruoyu A0281205N

