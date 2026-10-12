# House Price Prediction Model

**IT3091 Machine Learning · Group 2026-DS-48 · Guided Data Track · Domain 8: Real Estate**

A regression model that gives a real estate agency a fast, explainable first valuation for a house in Ames, Iowa. We predict `SalePrice` (modelled as `log1p(SalePrice)`) and report errors in USD.

## Team

| Part | Member | Responsibility |
|---|---|---|
| A | IT24100119 — Gangoda E.W.W.M.C.R.B. | Data understanding, data dictionary, missing values, cleaning, EDA, insight log |
| B | IT24100871 — Senevirathne J.A.D.T. | Split, missing-value strategy, outliers, feature engineering and reduction, encoding, scaling, pipeline |
| C | IT24101843 — Dharmadasa K.A.Y.T.H. | Baseline and alternative models, cross-validation, GridSearchCV tuning, model comparison log |
| D | IT24101829 — Ranasinghe R.P.V.K. | Final test evaluation, residuals, error diagnostics, feature importance, metrics logging |

## Dataset

Kaggle, *House Prices: Advanced Regression Techniques* (based on Dean De Cock's Ames Housing data). One row is one house sale in Ames, Iowa, 2006–2010.

| File | Rows × columns | Description |
|---|---|---|
| `train.csv` | 1460 × 81 | Labelled sales, including `SalePrice` |
| `test.csv` | 1459 × 80 | Kaggle test file, no `SalePrice` (used only for the final submission) |

## Repository contents

```
.
├── Ames_Housing_Price_Prediction.ipynb   # full pipeline with saved outputs
├── train.csv                             # 1,460 labelled sales
├── test.csv                              # Kaggle test file (no SalePrice)
├── requirements.txt
└── README.md
```

Running the notebook also creates a `figures/` folder (charts) and `submission.csv` (Kaggle predictions).

## How to run

1. Clone the repository and open the folder.
2. Install the libraries: `pip install -r requirements.txt`
3. Start Jupyter: `jupyter notebook`
4. Open `Ames_Housing_Price_Prediction.ipynb` and choose **Run All**. `train.csv` and `test.csv` must be in the same folder as the notebook.

Or open it in Google Colab, upload the notebook and both CSV files, and run all cells.

## Method in brief

- **Split:** stratified 70/15/15 on price deciles (1,022 train, 219 validation, 219 test). Validation is used for every choice; the test split is scored once.
- **No leakage:** only row-by-row fixes happen before the split; medians, modes, scalers and encoders are fitted on the training split only.
- **Features:** 9 engineered features; reduction from 88 to 62 features, each drop checked on validation (near-constant, |r| > 0.8 pairs, weak features). VarianceThreshold(0.01) is used; PCA was tested and rejected.
- **Models:** Linear Regression (baseline), Ridge, Random Forest, Gradient Boosting, tuned with GridSearchCV (5-fold CV on the training split).
- **Final model:** 50/50 ensemble of tuned Ridge and tuned Gradient Boosting.

## Results

| Split | RMSE (USD) | MAE (USD) | R² (log) |
|---|---|---|---|
| Validation | 18,615 | 12,904 | 0.9176 |
| Test (scored once) | 21,673 | 13,981 | 0.9049 |

Errors are largest for the most expensive homes (in dollars) and the cheapest homes (in percent), so those should go to a human appraiser.

## Limitations and responsible use

One city and period (2006–2010), small data, and a log target that slightly under-predicts average price. `Neighborhood` can act as a proxy for income, so check errors by area and keep a human appraiser in the loop. See the notebook's final section for the full list.

## Reproducibility

Random seed 42 is used everywhere. Library versions used: pandas 3.0.6, scikit-learn 1.9.1.

## Data source and licence

Data from Kaggle's House Prices competition (Ames Housing dataset, Dean De Cock, 2011). Use it under the competition's terms.
