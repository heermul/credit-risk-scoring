# Credit Risk Scoring Engine & Explainable Underwriting API

An end-to-end machine learning credit risk scoring system trained on **300,000+ LendingClub loans** to predict borrower default probability at loan origination, paired with a cost-sensitive decision threshold and FCRA-compliant **SHAP** adverse action reason codes.

## Key Highlights & Business Impact
* **Leakage Prevention:** Audited 151 raw attributes and removed 30+ post-origination variables (`recoveries`, `total_pymnt`, `last_fico_range_high/low`), preventing artificial validation inflation (`0.9388` leaky AUC $\rightarrow$ `0.7188` honest baseline).
* **Model Performance:** Trained a leak-free Scikit-Learn `ColumnTransformer` + **LightGBM** pipeline handling an **80/20 class imbalance**, achieving:
  * **ROC-AUC:** `0.7282` (vs `0.7188` Logistic Regression baseline)
  * **PR-AUC:** `0.3989`
  * **Gini Coefficient:** `0.4564`
  * **KS-Statistic:** `33.13%`
* **Cost-Sensitive Threshold Optimization:** Built an asymmetric dollar-loss matrix (`$10,000` principal loss per missed default vs `$1,500` missed interest per rejected good borrower). Shifting the decision cutoff from `0.50` to **`0.35`** improved default recall to **87.6%** and saved **$5.86M** across the 60,000-loan test portfolio.
* **Explainable AI (SHAP):** Integrated `shap.TreeExplainer` into the inference API to return the top-3 risk drivers for every rejected loan application.

## Tech Stack
* **Modeling & Explainability:** Python, LightGBM, Scikit-Learn, Pandas, NumPy, SHAP
* **Backend & Deployment:** FastAPI, Pydantic, Uvicorn, Docker, Hugging Face Spaces

## Project Structure
```text
├── app/
│   └── main.py              # FastAPI inference endpoint + SHAP reason codes
├── models/
│   └── credit_risk_model.joblib
├── notebooks/
│   └── 01_credit_risk_modeling.ipynb
├── Dockerfile
├── requirements.txt
└── README.md

## API Usage (`POST /predict`)
Send a JSON payload with applicant details to receive the underwriting decision, default probability, risk tier, and adverse action reasons:

```json
{
  "loan_amnt": 15000,
  "term": 60,
  "int_rate": 18.5,
  "annual_inc": 45000,
  "dti": 28.4,
  "fico_range_low": 660,
  "emp_length": 2,
  "grade": "D",
  "home_ownership": "RENT",
  "purpose": "debt_consolidation"
}