"""
NexaVault Financial Corp — Intelligent Loan Approval System
Version : 3.0.0  (fixed data generation & feature engineering)
Author  : Mandeep Sharma
Copyright (c) 2026 Mandeep Sharma. All rights reserved.
"""
import os, warnings, pathlib
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              precision_score, recall_score, classification_report)

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR   = pathlib.Path(__file__).parent.resolve()
COPYRIGHT  = "Copyright (c) 2026 Mandeep Sharma. All rights reserved."
COMPANY    = "NexaVault Financial Corp"

# ── Encoding maps ─────────────────────────────────────────────────────────
CREDIT_SCORE_MAP = {"Exceptional (800-850)":825,"Very Good (740-799)":770,
    "Good (670-739)":705,"Fair (580-669)":625,"Poor (500-579)":540,
    "Very Poor (300-499)":420,"No History":300}
EDU_MAP  = {"No Formal Education":0,"High School":1,"Associate Degree":2,
    "Bachelor's Degree":3,"Master's Degree":4,"Doctoral Degree":5}
EMP_MAP  = {"Full-time Employee":0,"Government/Military":0,"Retired":1,
    "Business Owner":1,"Self-Employed":2,"Independent Contractor":2,
    "Part-time Employee":2,"Freelancer":3,"Unemployed":4}
TEN_MAP  = {"<6 months":0,"6-12 months":1,"1-2 years":2,
    "2-5 years":3,"5-10 years":4,"10+ years":5}
BNK_MAP  = {"None":0,"Discharged 7+ years ago":1,"Discharged 2-7 years ago":2,
    "Discharged under 2 years":3,"Active/Pending":4}
COLL_MAP = {"Real Estate":3,"Investment Portfolio":2,"Business Assets":2,
    "Vehicle":1,"None (Unsecured)":0}
LTYP_MAP = {"Home Purchase":0,"Home Refinance":0,"HELOC":1,"Auto Loan":1,
    "Education Loan":1,"Debt Consolidation":2,"Business Loan":2,"Personal Loan":2}

NUM_FEATURES = ["Credit_Score_Num","Education_Num","Employment_Risk","Tenure_Num",
    "Bankruptcy_Num","Credit_Utilization_Pct","Late_Payments_2yr","Down_Payment_Pct",
    "Dependents","Total_Annual_Income","EMI","Income_EMI_Ratio","Loan_To_Income",
    "DTI_Ratio","Log_Income","Log_Loan","Net_Monthly_Income","Affordability_Score",
    "Loan_Amount_K","Loan_Term_Months","Collateral_Score","Loan_Type_Risk"]
CAT_FEATURES = ["Gender","Marital_Status","Loan_Purpose","Property_Area","Industry"]
DROP_COLS    = ["Loan_ID","Loan_Status","Target","Credit_Tier","Employment_Status",
    "Job_Tenure","Education","Bankruptcy_History","Loan_Type","Collateral","Credit_Score"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Credit_Tier" in d.columns:
        d["Credit_Tier"] = d["Credit_Tier"].astype(str).str.replace("\u2013","-").str.replace("\u2014","-")
    d["Credit_Score_Num"]     = d["Credit_Tier"].map(CREDIT_SCORE_MAP).fillna(300)
    d["Education_Num"]        = d["Education"].map(EDU_MAP).fillna(2)
    d["Employment_Risk"]      = d["Employment_Status"].map(EMP_MAP).fillna(2)
    d["Tenure_Num"]           = d["Job_Tenure"].map(TEN_MAP).fillna(2)
    d["Bankruptcy_Num"]       = d["Bankruptcy_History"].map(BNK_MAP).fillna(0)
    d["Collateral_Score"]     = d["Collateral"].map(COLL_MAP).fillna(1)
    d["Loan_Type_Risk"]       = d["Loan_Type"].map(LTYP_MAP).fillna(1)
    d["Total_Annual_Income"]  = (d["Applicant_Annual_Income"]
                                 + d.get("CoApplicant_Annual_Income", pd.Series([0]*len(d))).fillna(0)
                                 + d.get("Other_Income", pd.Series([0]*len(d))).fillna(0))
    d["Total_Monthly_Income"] = d["Total_Annual_Income"] / 12
    d["EMI"]                  = (d["Loan_Amount_K"].astype(float)*1000) / d["Loan_Term_Months"].replace(0,360)
    d["Income_EMI_Ratio"]     = d["Total_Monthly_Income"] / d["EMI"].replace(0,1)
    d["Loan_To_Income"]       = (d["Loan_Amount_K"].astype(float)*1000) / d["Total_Annual_Income"].replace(0,1)
    d["DTI_Ratio"]            = (d.get("Monthly_Debt_Obligations", pd.Series([0]*len(d))).fillna(0)
                                 + d["EMI"]) / d["Total_Monthly_Income"].replace(0,1)
    d["Log_Income"]           = np.log1p(d["Total_Annual_Income"])
    d["Log_Loan"]             = np.log1p(d["Loan_Amount_K"].astype(float))
    d["Net_Monthly_Income"]   = (d["Total_Monthly_Income"]
                                 - d.get("Monthly_Debt_Obligations", pd.Series([0]*len(d))).fillna(0))
    d["Affordability_Score"]  = d["Net_Monthly_Income"] / d["EMI"].replace(0,1)
    return d


def build_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),
                          ("scl",StandardScaler())]), NUM_FEATURES),
        ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                          ("ohe",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]), CAT_FEATURES),
    ])


def train_and_evaluate(X_train, X_test, y_train, y_test):
    pre = build_preprocessor()
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42),
        "SVM":                 SVC(kernel="rbf", probability=True, C=2.0, gamma="scale", random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results, best_auc, best_name, best_model = {}, 0, "", None

    print("\n=== 5-Fold Cross-Validation (ROC-AUC) ===")
    for name, clf in candidates.items():
        pipe  = Pipeline([("pre",pre),("clf",clf)])
        scores= cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        yp    = pipe.predict(X_test)
        ypr   = pipe.predict_proba(X_test)[:,1]
        auc   = roc_auc_score(y_test, ypr)
        results[name] = {"pipe":pipe,"cv":scores.mean(),"auc":auc,
            "f1":f1_score(y_test,yp),"acc":accuracy_score(y_test,yp),
            "prec":precision_score(y_test,yp),"rec":recall_score(y_test,yp)}
        print(f"  {name:<26} CV={scores.mean():.4f}  AUC={auc:.4f}  F1={results[name]['f1']:.4f}")
        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, pipe

    print(f"\n  Best: {best_name} (AUC={best_auc:.4f})")
    print(classification_report(y_test, best_model.predict(X_test), target_names=["Rejected","Approved"]))
    return results, best_name, best_model


def predict(model, applicant: dict) -> dict:
    df   = engineer_features(pd.DataFrame([applicant]))
    prob = float(model.predict_proba(df)[0,1])
    dec  = "Approved" if prob>=0.65 else ("Manual Review" if prob>=0.45 else "Rejected")
    risk = ("Very Low" if prob>=0.80 else "Low" if prob>=0.65 else
            "Medium" if prob>=0.45 else "High" if prob>=0.30 else "Very High")
    emi  = round((float(applicant["Loan_Amount_K"])*1000)/float(applicant["Loan_Term_Months"]),2)
    return {"decision":dec,"approval_probability":round(prob*100,2),"risk_level":risk,"monthly_emi":emi}


def main():
    print(f"\n{'='*65}")
    print(f"  {COMPANY} — Intelligent Loan Approval System v3.0")
    print(f"  {COPYRIGHT}")
    print(f"{'='*65}\n")

    data_path = BASE_DIR / "data" / "train_applications.csv"
    df = pd.read_csv(data_path)
    print(f"[NexaVault] Loaded {len(df):,} records from {data_path}")

    df = engineer_features(df)
    df["Target"] = (df["Loan_Status"]=="Approved").astype(int)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df["Target"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    results, best_name, best_model = train_and_evaluate(X_train,X_test,y_train,y_test)

    model_path = BASE_DIR / "app" / "nexavault_model.pkl"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"\n  Model saved → {model_path}")
    print(f"\n[NexaVault] Pipeline complete. System ready.\n")


if __name__ == "__main__":
    main()
