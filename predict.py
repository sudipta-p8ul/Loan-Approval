# ========================
# LOAN APPROVAL PREDICTION
# ========================

import pandas as pd
import joblib

# =========================
# 1. Load Saved Model Files
# =========================
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

print("\n========== LOAN APPROVAL PREDICTION ==========\n")

# =========================
# 2. Take User Input
# =========================
no_of_dependents = int(input("Enter number of dependents: "))
education = input("Education (Graduate / Not Graduate): ").strip()
self_employed = input("Self Employed (Yes / No): ").strip()
income_annum = float(input("Annual Income: "))
loan_amount = float(input("Loan Amount: "))
loan_term = float(input("Loan Term (in months): "))
cibil_score = float(input("CIBIL Score: "))
residential_assets_value = float(input("Residential Assets Value: "))
commercial_assets_value = float(input("Commercial Assets Value: "))
luxury_assets_value = float(input("Luxury Assets Value: "))
bank_asset_value = float(input("Bank Asset Value: "))

# =========================
# 3. Create DataFrame
# =========================
user_data = pd.DataFrame({
    "no_of_dependents": [no_of_dependents],
    "education": [education],
    "self_employed": [self_employed],
    "income_annum": [income_annum],
    "loan_amount": [loan_amount],
    "loan_term": [loan_term],
    "cibil_score": [cibil_score],
    "residential_assets_value": [residential_assets_value],
    "commercial_assets_value": [commercial_assets_value],
    "luxury_assets_value": [luxury_assets_value],
    "bank_asset_value": [bank_asset_value]
})

# One-hot encoding
user_data = pd.get_dummies(user_data)

# Match training columns
user_data = user_data.reindex(columns=model_columns, fill_value=0)

# Scale input
user_scaled = scaler.transform(user_data)

# =========================
# 4. Predict
# =========================
prediction = model.predict(user_scaled)[0]
probability = model.predict_proba(user_scaled)[0][1]

# =========================
# 5. Output Result
# =========================
print("\n===============================")

if prediction == 1:
    print("✅ Loan Approved")
else:
    print("❌ Loan Rejected")

print(f"Approval Probability: {probability:.4f}")
print("===============================\n")