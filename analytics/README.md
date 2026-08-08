# /analytics

End-to-end analytics pipeline on the Titanic dataset: EDA, missing-value handling,
univariate/bivariate/multivariate analysis, then predictive modeling (Logistic
Regression, Decision Tree, Random Forest) with hyperparameter tuning and imbalance
handling.

## Setup
pip install -r requirements.txt (pandas, numpy, seaborn, matplotlib, scikit-learn, joblib)

## Run
Open and run `analytics_pipeline.ipynb` top to bottom in Jupyter/Colab.

## Design decisions
- Missing values handled by a threshold rule: <5% missing -> drop rows, 5-30% ->
  impute (median/mode), >30% -> drop column.
- Train/test split happens before any preprocessing to avoid data leakage; stratified
  on `survived` due to class imbalance (~38%/62%).
- Preprocessing (imputing, scaling, encoding) is wrapped in a scikit-learn Pipeline +
  ColumnTransformer so it's fit only on training data.
- Random Forest hyperparameters tuned via GridSearchCV (5-fold CV, F1 scoring).
- Outputs (charts, best model) are saved as supporting artifacts alongside the notebook.
