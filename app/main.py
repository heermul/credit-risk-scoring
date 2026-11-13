import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

# 1. Saved model artifact load karte hain
artifact = joblib.load("models/credit_risk_model.joblib")
pipeline = artifact["pipeline"]
THRESHOLD = artifact["optimal_threshold"]  # 0.35
FEATURE_NAMES = artifact["feature_names"]
NUM_COLS = artifact["num_cols"]
CAT_COLS = artifact["cat_cols"]
DEFAULT_VALUES = artifact["default_values"]

# SHAP explainer initialize karte hain
preprocessor = pipeline.named_steps["preprocessor"]
lgb_model = pipeline.named_steps["classifier"]
explainer = shap.TreeExplainer(lgb_model)

app = FastAPI(
    title="LendingClub Credit Risk Scoring API",
    description="Predicts loan default risk with cost-optimal threshold (0.35) and FCRA-compliant SHAP reason codes.",
    version="1.0.0"
)

# 2. Pydantic Input Schema (Edge-case validation ke saath)
class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., gt=0, le=50000, example=15000)
    term: int = Field(..., description="36 or 60 months", example=36)
    int_rate: float = Field(..., gt=0, le=40, example=13.5)
    annual_inc: float = Field(..., gt=0, example=65000)
    dti: float = Field(..., ge=0, le=100, example=18.5)
    fico_range_low: float = Field(..., ge=300, le=850, example=690)
    emp_length: float = Field(default=5.0, ge=0, le=10)
    grade: str = Field(default="B", example="B")
    home_ownership: str = Field(default="MORTGAGE", example="MORTGAGE")
    purpose: str = Field(default="debt_consolidation", example="debt_consolidation")

@app.get("/")
def health_check():
    return {"status": "healthy", "optimal_cutoff": THRESHOLD}

@app.post("/predict")
def predict_default(application: LoanApplication):
    # 3. Raw input ko full 69-feature DataFrame mein map karte hain
    input_data = DEFAULT_VALUES.copy()
    for col in CAT_COLS:
        if col not in input_data:
            input_data[col] = "missing"
            
    # User ke diye gaye main fields update karte hain
    user_dict = application.model_dump()
    user_dict["fico_range_high"] = user_dict["fico_range_low"] + 4
    input_data.update(user_dict)
    
    df_input = pd.DataFrame([input_data])[NUM_COLS + CAT_COLS]
    
    # 4. Default probability aur business decision
    default_prob = float(pipeline.predict_proba(df_input)[0, 1])
    decision = "REJECTED" if default_prob >= THRESHOLD else "APPROVED"
    
    if default_prob < 0.20:
        risk_tier = "Low Risk"
    elif default_prob < THRESHOLD:
        risk_tier = "Moderate Risk"
    else:
        risk_tier = "High Risk"
        
    # 5. Top 3 SHAP Adverse Action Reason Codes nikalte hain
    X_transformed = preprocessor.transform(df_input)
    shap_vals = explainer.shap_values(X_transformed)
    vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
    
    shap_series = pd.Series(vals, index=FEATURE_NAMES)
    top_factors = shap_series.sort_values(ascending=False).head(3)
    
    reasons = [
        f"{feat} (+{score:.3f} risk impact)"
        for feat, score in top_factors.items() if score > 0
    ]
    
    return {
        "decision": decision,
        "default_probability": round(default_prob, 4),
        "optimal_threshold": THRESHOLD,
        "risk_tier": risk_tier,
        "adverse_action_reasons": reasons if decision == "REJECTED" else []
    }