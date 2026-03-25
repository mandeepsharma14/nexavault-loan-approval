# Changelog — NexaVault Intelligent Loan Approval System
Copyright (c) 2026 Mandeep Sharma. All rights reserved.

---

## [2.0.0] — 2026-03-24

### Added
- Expanded applicant profile: 7 gender identity options, 6 marital statuses
- Full Experian credit model with 7 tiers (Exceptional → No History)
- 10 employment status categories (W-2, contractor, freelancer, self-employed, etc.)
- Non-linear income slider (31 steps, calibrated to real income distributions)
- Co-applicant income, other income sources, monthly debt obligations
- DTI (Debt-to-Income) ratio as a key engineered feature
- Affordability Score feature
- Job tenure and industry sector as ML signals
- Bankruptcy history, late payment count, credit utilization fields
- Loan type (8 options), loan purpose, collateral, down payment fields
- Three-tier decision: Approved / Manual Review / Rejected
- Flask REST API with CORS support
- Docker + docker-compose deployment
- Full test suite

### Improved
- Model AUC improved from 93.4% to 97.4%
- Feature count increased from 12 raw → 20 engineered features
- More realistic synthetic data generation

---

## [1.0.0] — 2026-01-15

### Initial Release
- Basic loan approval prediction
- 4 model candidates (LR, RF, GBM, SVM)
- Simple Flask API
- Jupyter notebook
