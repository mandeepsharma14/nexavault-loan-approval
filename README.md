<div align="center">

# 🏦 NexaVault Financial Corp
## Intelligent Loan Approval System
### Powered by Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?logo=flask)
![License](https://img.shields.io/badge/License-Proprietary-red)

**Copyright (c) 2026 Mandeep Sharma. All rights reserved.**

*An end-to-end ML system that replaces slow, biased manual loan reviews with fast, accurate, data-driven decisions.*

</div>

---

## Overview

NexaVault's Intelligent Loan Approval System uses historical application data to predict loan approval outcomes with **97%+ ROC-AUC accuracy**. It replaces inconsistent manual officer review with a fair, explainable, and auditable ML pipeline.

### Key Outcomes

| Problem | Solution | Result |
|---------|----------|--------|
| Manual review takes 3–5 days | Instant ML prediction | Decision in < 200ms |
| Biased/inconsistent officers | Algorithm-driven scoring | Standardized criteria |
| Good customers rejected | High recall model (94%) | Lost business reduced |
| High-risk customers approved | Precision-optimized (94%) | Financial loss reduced |

---

## Project Structure

```
nexavault-loan-approval/
├── 📂 data/
│   ├── train_applications.csv      # 800 historical loan applications
│   ├── test_applications.csv       # 200 unlabeled test applications
│   └── sample_submission.csv       # Submission format example
├── 📂 notebooks/
│   └── nexavault_loan_approval.ipynb  # Full EDA → Training → Inference notebook
├── 📂 app/
│   ├── app.py                      # Flask REST API
│   └── nexavault_model.pkl         # Trained model (generated after training)
├── 📂 tests/
│   └── test_api.py                 # API unit tests
├── 📂 docs/
│   ├── eda_plots.png               # EDA visualizations (generated)
│   ├── model_evaluation.png        # ROC & metrics charts (generated)
│   └── confusion_matrix.png        # Confusion matrix (generated)
├── loan_approval_system.py         # Main ML pipeline
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container configuration
├── docker-compose.yml              # Multi-service orchestration
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore rules
├── LICENSE                         # Proprietary license
└── README.md                       # This file
```

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/mandeep-sharma/nexavault-loan-approval.git
cd nexavault-loan-approval
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python loan_approval_system.py
```

### 3. Launch the API

```bash
cd app && python app.py
# API running at http://localhost:5000
```

### 4. Test a Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Marital_Status": "Married",
    "Applicant_Annual_Income": 95000,
    "CoApplicant_Annual_Income": 45000,
    "Credit_Tier": "Very Good (740-799)",
    "Loan_Amount_K": 250,
    "Loan_Term_Months": 360,
    "Employment_Status": "Full-time Employee",
    "Late_Payments_2yr": 0,
    "Bankruptcy_History": "None",
    "Credit_Utilization_Pct": 18,
    "Monthly_Debt_Obligations": 600,
    "Down_Payment_Pct": 20,
    "Property_Area": "Urban",
    "Loan_Type": "Home Purchase",
    "Collateral": "Real Estate"
  }'
```

### 5. Run the Notebook

```bash
cd notebooks && jupyter notebook nexavault_loan_approval.ipynb
```

---

## Dataset

| Column | Type | Description |
|--------|------|-------------|
| `Loan_ID` | string | Unique application identifier |
| `Gender` | categorical | Applicant gender identity |
| `Marital_Status` | categorical | Single / Married / Divorced etc. |
| `Dependents` | integer | Number of financial dependents |
| `Education` | categorical | Highest education level |
| `Employment_Status` | categorical | W-2 / Self-Employed / Contractor etc. |
| `Job_Tenure` | categorical | Length of employment |
| `Applicant_Annual_Income` | integer | Gross annual income ($) |
| `CoApplicant_Annual_Income` | integer | Co-applicant income ($) |
| `Credit_Tier` | categorical | Experian-model tier (Exceptional → Very Poor) |
| `Credit_Score` | integer | Numeric credit score (300–850) |
| `Late_Payments_2yr` | integer | Count of late payments in last 2 years |
| `Credit_Utilization_Pct` | integer | Credit card utilization (%) |
| `Bankruptcy_History` | categorical | Bankruptcy status |
| `Loan_Amount_K` | integer | Requested loan amount ($K) |
| `Loan_Term_Months` | integer | Repayment term in months |
| `Down_Payment_Pct` | float | Down payment percentage |
| `Loan_Status` | categorical | **Target** — Approved / Manual Review / Rejected |

---

## ML Pipeline

```
Raw Data → Feature Engineering → Preprocessing → Model Training → Evaluation → Deployment
```

**Feature Engineering** creates 14 derived features including:
- `Income_EMI_Ratio` — key affordability signal
- `DTI_Ratio` — debt-to-income ratio
- `Affordability_Score` — net income after debt / EMI
- `Credit_Score_Num` — numeric encoding of Experian tier
- Log transforms for income and loan amount

**Models Trained:**
- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- SVM (RBF kernel)

**Selection Criterion:** 5-fold stratified cross-validation ROC-AUC

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model status |
| `/predict` | POST | Single applicant prediction |
| `/predict/batch` | POST | CSV batch prediction |

**Response format:**
```json
{
  "decision": "Approved",
  "approval_probability": 87.3,
  "risk_level": "Low",
  "monthly_emi": 1243.56,
  "processed_at": "2026-03-24T10:30:00"
}
```

---

## Docker Deployment

```bash
# Build and run
docker build -t nexavault-app .
docker run -p 5000:5000 nexavault-app

# Or with docker-compose
docker-compose up --build
```

---

## License

**Proprietary — Copyright (c) 2026 Mandeep Sharma. All rights reserved.**

This software is the intellectual property of Mandeep Sharma. Unauthorized copying, distribution, or commercial use is strictly prohibited. See [LICENSE](LICENSE) for full terms.

---

<div align="center">
  <sub>Built with ❤️ by Mandeep Sharma · NexaVault Financial Corp · 2026</sub>
</div>
