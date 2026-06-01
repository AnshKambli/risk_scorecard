# 📊 Risk Scorecard Analyser

A Streamlit web application for borrower portfolio risk scoring, band classification, and model validation — built on a four-tier rule-based scorecard with logistic regression validation.

---

## 🚀 Live App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)][https://riskscorecard-4ngw9exuzj9jztbavdjjvo.streamlit.app/)]

---

## 🎯 What It Does

Upload a borrower portfolio CSV and instantly get:

- **Four-tier risk classification** — Low / Medium / High / Very High Risk
- **Rule-based scoring** across 9 borrower attributes
- **Logistic regression validation** with AUC, Gini, and KS statistics
- **WoE / IV analysis** for feature importance ranking
- **Interactive dashboard** with 7 charts
- **Downloadable reports** — scored CSV, band performance, WoE/IV table, dashboard PNG

---

## 🗂 Risk Band Thresholds

| Band | Score Range | Default Rate | Decision |
|------|-------------|--------------|----------|
| 🟢 Low Risk | 70 – 100 | ~0% | ✅ Approve |
| 🟡 Medium Risk | 45 – 69 | ~8% | 🔍 Review |
| 🟠 High Risk | 25 – 44 | ~32% | ⛔ Decline |
| 🔴 Very High Risk | 0 – 24 | ~58% | ⛔ Decline |

---

## 📋 Required Input Columns

Your CSV must contain these columns (exact names):

| Column | Description |
|--------|-------------|
| `Bank Lan` | Bank loan reference number |
| `Customer ID` | Unique customer identifier |
| `Document Collection Status (Yes/No)19-11-2025` | Document collection flag |
| `Mobile No. Status (Yes/No)_27-01-2025` | Mobile number verified flag |
| `Pan Card Status` | PAN card availability |
| `Voter ID Number Status` | Voter ID availability |
| `Universal ID Number Status` | Aadhaar/UID availability |
| `DL_ Status` | Driving licence availability |
| `Name Validation Final Remark` | Name match result |
| `Contactable/Non-Contactable` | Contact status |
| `CIBIL Data` | CIBIL bureau data availability |

---

## 📊 Scorecard Logic

| Attribute | Value | Points |
|-----------|-------|--------|
| Name Validation | Yes_Correct Customer | 20 |
| CIBIL Data | Yes | 20 |
| Document | Documents Collected | 15 |
| Mobile No. | Yes | 10 |
| Connected | connected | 10 |
| PAN Card | Yes | 7 |
| UID | Yes | 7 |
| Voter ID | Yes | 6 |
| DL | Yes | 5 |

**Maximum possible score: 100**

---

## 🧠 Model Validation

A logistic regression model trained on WoE-encoded features validates the scorecard with:

- **ROC-AUC** — discriminatory power
- **Gini coefficient** — (2 × AUC) − 1
- **KS statistic** — maximum separation between Good and Bad distributions

---

## 📦 Installation (Local)

```bash
git clone https://github.com/AnshKambli/risk-scorecard.git
cd risk-scorecard
pip install -r requirements.txt
streamlit run app.py
```

---

## 🗃 Project Structure

```
risk_scorecard/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## ⬇️ Output Files

| File | Contents |
|------|----------|
| `scored_output.csv` | All records with score, risk band, and target |
| `band_performance.csv` | Count, default rate, avg score per band |
| `woe_iv_table.csv` | WoE and IV values per feature category |
| `risk_dashboard.png` | Full 7-panel dashboard image |

---

## 🔧 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-blue)

---

## 👤 Author

**Ansh Kambli**  
Data Analyst — Reliance Asset Reconstruction Company  
[GitHub](https://github.com/AnshKambli) · [Portfolio](https://anshkambli.github.io/Portfolio)
