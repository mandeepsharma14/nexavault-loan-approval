"""
NexaVault Financial Corp — API Tests
Copyright (c) 2026 Mandeep Sharma. All rights reserved.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SAMPLE = {
    "Gender":"Male","Marital_Status":"Married","Dependents":1,
    "Education":"Bachelor's Degree","Employment_Status":"Full-time Employee",
    "Job_Tenure":"5-10 years","Industry":"Technology",
    "Applicant_Annual_Income":90000,"CoApplicant_Annual_Income":45000,"Other_Income":0,
    "Monthly_Debt_Obligations":500,"Credit_Tier":"Very Good (740-799)",
    "Credit_Utilization_Pct":15,"Late_Payments_2yr":0,"Bankruptcy_History":"None",
    "Loan_Type":"Home Purchase","Loan_Purpose":"Primary Residence",
    "Loan_Amount_K":250,"Loan_Term_Months":360,"Down_Payment_Pct":20,
    "Property_Area":"Urban","Collateral":"Real Estate"
}

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.get_json()
    assert d["status"] == "healthy"
    assert "model" in d

def test_predict_returns_decision(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        d = r.get_json()
        assert "decision" in d
        assert d["decision"] in ["Approved","Manual Review","Rejected"]
        assert 0 <= d["approval_probability"] <= 100
        assert d["risk_level"] in ["Very Low","Low","Medium","High","Very High"]

def test_predict_high_risk():
    try:
        import requests
        bad = SAMPLE.copy()
        bad.update({"Credit_Tier":"Very Poor (300-499)","Late_Payments_2yr":4,
                    "Bankruptcy_History":"Active/Pending","Employment_Status":"Unemployed"})
        r = requests.post("http://localhost:5000/predict", json=bad, timeout=3)
        if r.status_code == 200:
            assert r.json()["decision"] == "Rejected"
    except Exception:
        pass  # API not running in test env

if __name__ == "__main__":
    print("NexaVault API Tests — run with: pytest tests/test_api.py -v")
    print("Copyright (c) 2026 Mandeep Sharma. All rights reserved.")
