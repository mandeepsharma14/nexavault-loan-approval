# NexaVault Loan Approval System — Personal Cheatsheet
## Mandeep Sharma · Copyright (c) 2026

---

## ENVIRONMENT SETUP

```bash
# Clone repo
git clone https://github.com/mandeep-sharma/nexavault-loan-approval.git
cd nexavault-loan-approval

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## TRAIN THE MODEL

```bash
# From project root — trains all 4 models, saves best to app/nexavault_model.pkl
python loan_approval_system.py

# Expected output:
#   Logistic Regression  CV-AUC: 0.9822
#   Random Forest        CV-AUC: 0.9205
#   Gradient Boosting    CV-AUC: 0.9287
#   SVM                  CV-AUC: 0.9800
#   Best: SVM (AUC=0.9736)
#   Model saved → app/nexavault_model.pkl
```

---

## RUN THE JUPYTER NOTEBOOK

```bash
# Option A — Classic Jupyter
cd notebooks
jupyter notebook nexavault_loan_approval.ipynb

# Option B — JupyterLab
jupyter lab notebooks/nexavault_loan_approval.ipynb

# Option C — VS Code
# Open nexavault_loan_approval.ipynb directly in VS Code
# Select kernel: Python 3 (venv)

# Run all cells at once (headless)
jupyter nbconvert --to notebook --execute notebooks/nexavault_loan_approval.ipynb
```

---

## RUN THE FLASK API

```bash
# Development mode
cd app
python app.py
# Listening on http://0.0.0.0:5000

# Production mode (gunicorn)
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Check health
curl http://localhost:5000/health
```

---

## API CALLS (curl)

```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Marital_Status": "Married",
    "Dependents": 1,
    "Education": "Master'\''s Degree",
    "Employment_Status": "Full-time Employee",
    "Job_Tenure": "5-10 years",
    "Industry": "Technology",
    "Applicant_Annual_Income": 120000,
    "CoApplicant_Annual_Income": 60000,
    "Other_Income": 5000,
    "Monthly_Debt_Obligations": 500,
    "Credit_Tier": "Exceptional (800-850)",
    "Credit_Utilization_Pct": 8,
    "Late_Payments_2yr": 0,
    "Bankruptcy_History": "None",
    "Loan_Type": "Home Purchase",
    "Loan_Purpose": "Primary Residence",
    "Loan_Amount_K": 350,
    "Loan_Term_Months": 360,
    "Down_Payment_Pct": 20,
    "Property_Area": "Urban",
    "Collateral": "Real Estate"
  }'

# Expected response:
# {
#   "decision": "Approved",
#   "approval_probability": 94.7,
#   "risk_level": "Very Low",
#   "monthly_emi": 972.22
# }
```

---

## DOCKER COMMANDS

```bash
# Build image
docker build -t nexavault-app .

# Run container
docker run -d -p 5000:5000 --name nexavault nexavault-app

# Docker Compose (recommended)
docker-compose up --build -d

# View logs
docker logs nexavault -f

# Stop
docker-compose down

# Rebuild after code change
docker-compose up --build -d
```

---

## PYTHON INFERENCE (script)

```python
import joblib, pandas as pd, numpy as np

model = joblib.load("app/nexavault_model.pkl")

CREDIT_SCORE_MAP = {
    "Exceptional (800-850)":825, "Very Good (740-799)":770,
    "Good (670-739)":705, "Fair (580-669)":625,
    "Poor (500-579)":540, "Very Poor (300-499)":420, "No History":0
}

def predict(applicant: dict) -> dict:
    df = pd.DataFrame([applicant])
    # Feature engineering
    df["Credit_Score_Num"]    = df["Credit_Tier"].map(CREDIT_SCORE_MAP).fillna(0)
    df["Education_Num"]       = 3   # default: Bachelor's
    df["Employment_Risk"]     = 0   # default: Full-time
    df["Tenure_Num"]          = 3   # default: 2-5 years
    df["Bankruptcy_Num"]      = 0
    df["Total_Annual_Income"] = df["Applicant_Annual_Income"] + df.get("CoApplicant_Annual_Income", pd.Series([0])).fillna(0)
    df["Total_Monthly_Income"]= df["Total_Annual_Income"] / 12
    df["EMI"]                 = (df["Loan_Amount_K"] * 1000) / df["Loan_Term_Months"]
    df["Income_EMI_Ratio"]    = df["Total_Monthly_Income"] / df["EMI"].replace(0,1)
    df["Loan_To_Income"]      = (df["Loan_Amount_K"] * 1000) / df["Total_Annual_Income"].replace(0,1)
    df["DTI_Ratio"]           = (df.get("Monthly_Debt_Obligations", pd.Series([0])).fillna(0) + df["EMI"]) / df["Total_Monthly_Income"].replace(0,1)
    df["Log_Income"]          = np.log1p(df["Total_Annual_Income"])
    df["Log_Loan"]            = np.log1p(df["Loan_Amount_K"])
    df["Net_Monthly_Income"]  = df["Total_Monthly_Income"] - df.get("Monthly_Debt_Obligations", pd.Series([0])).fillna(0)
    df["Affordability_Score"] = df["Net_Monthly_Income"] / df["EMI"].replace(0,1)

    prob = float(model.predict_proba(df)[0, 1])
    decision = "Approved" if prob >= 0.65 else ("Manual Review" if prob >= 0.45 else "Rejected")
    return {"decision": decision, "probability": round(prob * 100, 2)}
```

---

## QUICK MODEL RE-TRAIN (Python)

```python
from loan_approval_system import load_data, engineer_features, build_preprocessor
from loan_approval_system import train_models, evaluate_models, save_model
import pandas as pd
from sklearn.model_selection import train_test_split

df = load_data("data/train_applications.csv")
df = engineer_features(df)
df["Target"] = (df["Loan_Status"] == "Approved").astype(int)

drop = ["Loan_ID","Loan_Status","Target","Credit_Tier","Employment_Status",
        "Job_Tenure","Education","Bankruptcy_History"]
X = df.drop(columns=drop)
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                      stratify=y, random_state=42)
pre = build_preprocessor()
results = train_models(X_train, y_train, pre)
results, best_name, best_model = evaluate_models(results, X_test, y_test)
save_model(best_model)
```

---

## RUN TESTS

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_api.py::test_predict_returns_decision -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

---

## KEY FILE PATHS

| File | Purpose |
|------|---------|
| `loan_approval_system.py` | Full ML training pipeline |
| `app/app.py` | Flask REST API server |
| `app/nexavault_model.pkl` | Trained model (generated) |
| `notebooks/nexavault_loan_approval.ipynb` | Interactive notebook |
| `data/train_applications.csv` | 800 training records |
| `data/test_applications.csv` | 200 unlabeled test records |
| `Dockerfile` | Container build instructions |
| `docker-compose.yml` | Multi-service orchestration |

---

## EXPERIAN CREDIT TIER REFERENCE

| Tier | Score Range | Approval Impact |
|------|-------------|-----------------|
| Exceptional | 800–850 | Strong approval signal |
| Very Good | 740–799 | Above-average odds |
| Good | 670–739 | Standard approval |
| Fair | 580–669 | Higher rates, may need co-signer |
| Poor | 500–579 | Limited options |
| Very Poor | 300–499 | Very high rejection risk |
| No History | N/A | Alternative verification needed |

---

## DECISION THRESHOLDS

| Probability | Decision |
|-------------|----------|
| ≥ 65% | ✅ Approved |
| 45%–64% | ⏸ Manual Review |
| < 45% | ❌ Rejected |

---

## TROUBLESHOOTING

```bash
# Model file not found
python loan_approval_system.py   # Re-train

# Port 5000 already in use
lsof -i :5000 | awk 'NR>1{print $2}' | xargs kill -9
python app/app.py

# ModuleNotFoundError
pip install -r requirements.txt

# Docker permission denied
sudo docker build -t nexavault-app .

# Notebook kernel not found
python -m ipykernel install --user --name=venv
```

---

*Copyright (c) 2026 Mandeep Sharma. All rights reserved.*
*NexaVault Financial Corp — Intelligent Loan Approval System*
