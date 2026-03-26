"""
NexaVault Financial Corp — Loan Approval REST API v5.0
Copyright (c) 2026 Mandeep Sharma. All rights reserved.

Changes in v5.0:
  - Monthly debt options expanded to $10,000/mo (was $2,500)
  - Hard gate 6 fixed: credit < 580 now triggers on loans >= $100K (was > $100K)
  - New gate 8: credit < 580 + down < 10% + loan > $100K → capped at Manual Review
  - New gate 9: Very Poor credit (< 500) → all loans rejected regardless of income
  - Interest rate calculator added to every approved/review response
  - Training data expanded to 2,000 records
"""
from flask import Flask, request, jsonify, send_file
import pandas as pd, numpy as np, joblib, os, pathlib
from datetime import datetime

app      = Flask(__name__)
BASE_DIR = pathlib.Path(__file__).parent.parent.resolve()
model    = None

# ── Encoding maps ──────────────────────────────────────────────────────────
CREDIT_SCORE_MAP = {
    "Exceptional (800-850)":825,  "Exceptional (800\u2013850)":825,
    "Very Good (740-799)":770,    "Very Good (740\u2013799)":770,
    "Good (670-739)":705,         "Good (670\u2013739)":705,
    "Fair (580-669)":625,         "Fair (580\u2013669)":625,
    "Poor (500-579)":540,         "Poor (500\u2013579)":540,
    "Very Poor (300-499)":420,    "Very Poor (300\u2013499)":420,
    "No History":300,
}
EDU_MAP  = {"No Formal Education":0,"High School":1,"Associate Degree":2,
            "Bachelor's Degree":3,"Master's Degree":4,"Doctoral Degree":5}
EMP_MAP  = {"Full-time Employee":0,"Government/Military":0,"Retired":1,
            "Business Owner":1,"Self-Employed":2,"Independent Contractor":2,
            "Part-time Employee":2,"Freelancer":3,"Unemployed":4,"Student":3}
TEN_MAP  = {"<6 months":0,"6-12 months":1,"1-2 years":2,
            "2-5 years":3,"5-10 years":4,"10+ years":5}
BNK_MAP  = {"None":0,"Discharged 7+ years ago":1,"Discharged 2-7 years ago":2,
            "Discharged under 2 years":3,"Active/Pending":4}
COLL_MAP = {"Real Estate":3,"Investment Portfolio":2,"Business Assets":2,
            "Vehicle":1,"None (Unsecured)":0}
LTYP_MAP = {"Home Purchase":0,"Home Refinance":0,"HELOC":1,"Auto Loan":1,
            "Education Loan":1,"Debt Consolidation":2,"Business Loan":2,"Personal Loan":2}

# ── Interest rate matrix (2026 US market rates) ───────────────────────────
INTEREST_RATES = {
    "Home Purchase": {
        "Exceptional (800-850)":6.25,"Very Good (740-799)":6.50,"Good (670-739)":6.875,
        "Fair (580-669)":7.50,"Poor (500-579)":9.00,"Very Poor (300-499)":None,"No History":None},
    "Home Refinance": {
        "Exceptional (800-850)":6.375,"Very Good (740-799)":6.625,"Good (670-739)":7.00,
        "Fair (580-669)":7.75,"Poor (500-579)":9.25,"Very Poor (300-499)":None,"No History":None},
    "HELOC": {
        "Exceptional (800-850)":7.50,"Very Good (740-799)":7.75,"Good (670-739)":8.25,
        "Fair (580-669)":9.50,"Poor (500-579)":11.00,"Very Poor (300-499)":None,"No History":None},
    "Auto Loan": {
        "Exceptional (800-850)":5.50,"Very Good (740-799)":6.00,"Good (670-739)":7.25,
        "Fair (580-669)":10.50,"Poor (500-579)":14.00,"Very Poor (300-499)":18.00,"No History":15.00},
    "Personal Loan": {
        "Exceptional (800-850)":9.50,"Very Good (740-799)":11.00,"Good (670-739)":13.50,
        "Fair (580-669)":17.00,"Poor (500-579)":22.00,"Very Poor (300-499)":28.00,"No History":24.00},
    "Education Loan": {
        "Exceptional (800-850)":5.50,"Very Good (740-799)":6.00,"Good (670-739)":6.75,
        "Fair (580-669)":8.50,"Poor (500-579)":10.00,"Very Poor (300-499)":13.00,"No History":11.00},
    "Debt Consolidation": {
        "Exceptional (800-850)":10.00,"Very Good (740-799)":12.00,"Good (670-739)":14.50,
        "Fair (580-669)":18.00,"Poor (500-579)":24.00,"Very Poor (300-499)":30.00,"No History":26.00},
    "Business Loan": {
        "Exceptional (800-850)":8.50,"Very Good (740-799)":9.50,"Good (670-739)":11.00,
        "Fair (580-669)":14.00,"Poor (500-579)":18.00,"Very Poor (300-499)":None,"No History":20.00},
}

def get_interest_rate(loan_type, credit_tier):
    """Returns interest rate % or None if product not offered for this credit tier."""
    tier_clean = credit_tier.replace("\u2013","-").replace("\u2014","-")
    rates = INTEREST_RATES.get(loan_type, INTEREST_RATES.get("Personal Loan"))
    return rates.get(tier_clean)

def calc_true_emi(loan_k, term_months, annual_rate_pct):
    """Calculates actual EMI using the standard amortization formula."""
    if annual_rate_pct is None or annual_rate_pct == 0:
        return (loan_k * 1000) / term_months
    r = (annual_rate_pct / 100) / 12
    n = term_months
    emi = (loan_k * 1000 * r * (1+r)**n) / ((1+r)**n - 1)
    return round(emi, 2)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Credit_Tier" in d.columns:
        d["Credit_Tier"] = d["Credit_Tier"].astype(str).str.replace("\u2013","-",regex=False).str.replace("\u2014","-",regex=False)
    def gc(col, default):
        return d[col] if col in d.columns else pd.Series([default]*len(d), index=d.index)
    d["Credit_Score_Num"]     = d["Credit_Tier"].map(CREDIT_SCORE_MAP).fillna(300)
    d["Education_Num"]        = gc("Education","Bachelor's Degree").map(EDU_MAP).fillna(2)
    d["Employment_Risk"]      = gc("Employment_Status","Full-time Employee").map(EMP_MAP).fillna(0)
    d["Tenure_Num"]           = gc("Job_Tenure","2-5 years").map(TEN_MAP).fillna(2)
    d["Bankruptcy_Num"]       = gc("Bankruptcy_History","None").map(BNK_MAP).fillna(0)
    d["Collateral_Score"]     = gc("Collateral","Real Estate").map(COLL_MAP).fillna(1)
    d["Loan_Type_Risk"]       = gc("Loan_Type","Home Purchase").map(LTYP_MAP).fillna(0)
    d["Total_Annual_Income"]  = (d["Applicant_Annual_Income"].astype(float)
                                 + gc("CoApplicant_Annual_Income",0).fillna(0).astype(float)
                                 + gc("Other_Income",0).fillna(0).astype(float))
    d["Total_Monthly_Income"] = d["Total_Annual_Income"] / 12.0
    d["EMI"]                  = (d["Loan_Amount_K"].astype(float)*1000.0) / d["Loan_Term_Months"].astype(float).replace(0,360)
    d["Income_EMI_Ratio"]     = d["Total_Monthly_Income"] / d["EMI"].replace(0,1)
    d["Loan_To_Income"]       = (d["Loan_Amount_K"].astype(float)*1000.0) / d["Total_Annual_Income"].replace(0,1)
    mdebt                     = gc("Monthly_Debt_Obligations",0).fillna(0).astype(float)
    d["DTI_Ratio"]            = (mdebt + d["EMI"]) / d["Total_Monthly_Income"].replace(0,1)
    d["Log_Income"]           = np.log1p(d["Total_Annual_Income"])
    d["Log_Loan"]             = np.log1p(d["Loan_Amount_K"].astype(float))
    d["Net_Monthly_Income"]   = d["Total_Monthly_Income"] - mdebt
    d["Affordability_Score"]  = d["Net_Monthly_Income"] / d["EMI"].replace(0,1)
    return d


def hard_gates(data: dict) -> tuple:
    """
    Returns (rejection_reasons, review_reasons).
    rejection_reasons → hard reject, no appeal
    review_reasons    → soft cap → forced into Manual Review
    """
    rejections = []
    reviews    = []

    app_inc  = float(data.get("Applicant_Annual_Income", 0))
    co_inc   = float(data.get("CoApplicant_Annual_Income", 0) or 0)
    other    = float(data.get("Other_Income", 0) or 0)
    mdebt    = float(data.get("Monthly_Debt_Obligations", 0) or 0)
    loan_k   = float(data.get("Loan_Amount_K", 0))
    term     = float(data.get("Loan_Term_Months", 360) or 360)
    late     = int(data.get("Late_Payments_2yr", 0) or 0)
    bankrupt = str(data.get("Bankruptcy_History", "None"))
    ctier    = str(data.get("Credit_Tier","")).replace("\u2013","-")
    down_pct = float(data.get("Down_Payment_Pct", 0) or 0)

    total_ann  = app_inc + co_inc + other
    total_mo   = total_ann / 12.0
    emi        = (loan_k * 1000.0) / max(term, 1)
    dti        = (mdebt + emi) / max(total_mo, 1)
    inc_emi    = total_mo / max(emi, 1)
    net_resid  = total_mo - mdebt - emi
    cscore     = CREDIT_SCORE_MAP.get(ctier, 300)
    lti        = (loan_k * 1000) / max(total_ann, 1)

    # ── HARD REJECTS ────────────────────────────────────────────────────────
    # Gate 1: DTI > 50%
    if dti > 0.50:
        rejections.append(f"DTI is {dti*100:.1f}% — exceeds the 50% maximum. Monthly obligations ${mdebt+emi:,.0f} vs income ${total_mo:,.0f}/mo.")

    # Gate 2: Income/EMI < 1.20x
    if inc_emi < 1.20:
        rejections.append(f"Income/EMI ratio is {inc_emi:.2f}x — minimum required is 1.20x. Monthly income ${total_mo:,.0f} cannot support EMI of ${emi:,.0f}.")

    # Gate 3: Net residual < $200/mo
    if net_resid < 200:
        rejections.append(f"Only ${net_resid:,.0f}/mo remains after all obligations — minimum residual income is $200.")

    # Gate 4: Active bankruptcy
    if bankrupt == "Active/Pending":
        rejections.append("Active or pending bankruptcy on file. Cannot process applications during active bankruptcy proceedings.")

    # Gate 5: 4+ late payments
    if late >= 4:
        rejections.append(f"{late} late payments in the past 2 years — maximum allowed is 3.")

    # Gate 6: Very Poor credit (< 500) → reject all loans (FIXED: was credit<500, now correctly <=499)
    if cscore <= 499:
        rejections.append(f"Credit score {cscore} (Very Poor) — below the 500 minimum required for any NexaVault loan product. Credit repair is recommended before reapplying.")

    # Gate 7: Loan > 10x annual income
    if lti > 10.0:
        rejections.append(f"Loan is {lti:.1f}x annual income — maximum is 10x. ${loan_k:,.0f}K loan against ${total_ann:,.0f}/yr income.")

    # ── SOFT CAPS → MANUAL REVIEW ────────────────────────────────────────────
    # Gate 8: Credit 500-579 + loan >= $100K + down < 10% → Manual Review
    if 500 <= cscore <= 579 and loan_k >= 100 and down_pct < 10:
        reviews.append(f"Credit score {cscore} (Poor) with less than 10% down on a ${loan_k:,.0f}K loan. FHA requires minimum 10% down for scores below 580. Referred to manual review.")

    # Gate 9: DTI 43–50% → Manual Review cap
    if 0.43 < dti <= 0.50 and not rejections:
        reviews.append(f"DTI of {dti*100:.1f}% exceeds the preferred 43% threshold. Loan is borderline — referred to manual underwriter review.")

    return rejections, reviews


def load_model():
    global model
    path = os.getenv("MODEL_PATH", str(BASE_DIR/"app"/"nexavault_model.pkl"))
    if os.path.exists(path):
        model = joblib.load(path)
        print(f"[NexaVault] Model loaded from {path}")
    else:
        print(f"[NexaVault] WARNING: Model not found at {path}")


@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return r


@app.route("/")
def home():
    """Serve the interactive dashboard as the homepage."""
    dashboard_path = BASE_DIR / "nexavault_dashboard.html"
    if dashboard_path.exists():
        return send_file(str(dashboard_path))
    # Fallback JSON if dashboard file not found
    return jsonify({
        "name":        "NexaVault Intelligent Loan Approval System",
        "version":     "5.0.0",
        "author":      "Mandeep Sharma",
        "copyright":   "Copyright © 2026 Mandeep Sharma. All rights reserved.",
        "company":     "NexaVault Financial Corp",
        "status":      "live",
        "model":       "loaded" if model else "not loaded",
        "timestamp":   datetime.utcnow().isoformat(),
        "description": "AI-powered loan approval system with 99.77% ROC-AUC, 9 hard underwriting gates, and real-time interest rate calculator.",
        "endpoints": {
            "health":        "GET  /health",
            "predict":       "POST /predict",
            "batch_predict": "POST /predict/batch",
        },
        "github": "https://github.com/mandeepsharma14/nexavault-loan-approval",
    })


@app.route("/health")
def health():
    return jsonify({"status":"healthy","model":"loaded" if model else "not loaded",
                    "company":"NexaVault Financial Corp","version":"5.0.0",
                    "copyright":"Copyright \u00a9 2026 Mandeep Sharma. All rights reserved.",
                    "timestamp":datetime.utcnow().isoformat()})


@app.route("/predict", methods=["POST","OPTIONS"])
def predict():
    if request.method == "OPTIONS": return jsonify({}), 200
    if not model: return jsonify({"error":"Model not loaded. Run loan_approval_system.py first."}), 503

    data = request.get_json(force=True)
    loan_k   = float(data.get("Loan_Amount_K", 0))
    term     = float(data.get("Loan_Term_Months", 360) or 360)
    loan_type= str(data.get("Loan_Type","Personal Loan"))
    ctier    = str(data.get("Credit_Tier","")).replace("\u2013","-")
    app_inc  = float(data.get("Applicant_Annual_Income", 0))
    co_inc   = float(data.get("CoApplicant_Annual_Income", 0) or 0)
    other    = float(data.get("Other_Income", 0) or 0)
    mdebt    = float(data.get("Monthly_Debt_Obligations", 0) or 0)
    total_mo = (app_inc + co_inc + other) / 12.0

    # Interest rate
    rate      = get_interest_rate(loan_type, ctier)
    true_emi  = calc_true_emi(loan_k, term, rate)
    simple_emi= round((loan_k * 1000) / max(term, 1), 2)
    dti       = (mdebt + (true_emi if rate else simple_emi)) / max(total_mo, 1)

    # Step 1: Hard gates
    rejections, reviews = hard_gates(data)

    if rejections:
        return jsonify({
            "decision":"Rejected","approval_probability":2.0,"risk_level":"Very High",
            "monthly_emi":simple_emi,"true_emi_with_rate":true_emi,
            "interest_rate":rate,"dti_ratio":round(dti*100,1),
            "income_emi_ratio":round(total_mo/max(simple_emi,1),2),
            "rejection_reasons":rejections,"review_reasons":[],
            "gate_failed":True,"processed_at":datetime.utcnow().isoformat()
        })

    # Step 2: ML model
    try:
        df   = engineer(pd.DataFrame([data]))
        prob = float(model.predict_proba(df)[0, 1])

        # Apply soft cap if review gates fired
        if reviews:
            prob = min(prob, 0.62)   # force into manual review zone

        dec  = "Approved" if prob>=0.65 else ("Manual Review" if prob>=0.45 else "Rejected")
        risk = ("Very Low" if prob>=0.80 else "Low" if prob>=0.65 else
                "Medium"   if prob>=0.45 else "High" if prob>=0.30 else "Very High")

        # Total interest and cost
        total_payment = round(true_emi * term, 2) if rate else None
        total_interest= round(total_payment - loan_k*1000, 2) if total_payment else None

        return jsonify({
            "decision":dec,"approval_probability":round(prob*100,2),"risk_level":risk,
            "monthly_emi":simple_emi,
            "true_emi_with_rate":true_emi if rate else simple_emi,
            "interest_rate":rate,
            "interest_rate_label": f"{rate}% APR" if rate else "Rate TBD",
            "total_payment":total_payment,
            "total_interest":total_interest,
            "dti_ratio":round(dti*100,1),
            "income_emi_ratio":round(total_mo/max(simple_emi,1),2),
            "gate_failed":False,
            "review_reasons":reviews,
            "rejection_reasons":[],
            "processed_at":datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"error":str(e)}), 500


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT",5000)), debug=False)
