import streamlit as st
import joblib
import numpy as np
import os

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="EMI Eligibility Checker",
    page_icon="💳",
    layout="centered",
)

# ── Load Models ─────────────────────────────────────────
@st.cache_resource
def load_models():

    models = {}

    model_paths = {
        "classifier": "models/emi_classifier.pkl",
        "regressor": "models/emi_regressor.pkl"
    }

    for name, path in model_paths.items():

        if not os.path.exists(path):
            st.error(f"❌ Model not found: {path}")
            st.stop()

        models[name] = joblib.load(path)

    return models


models = load_models()
classifier = models["classifier"]
regressor = models["regressor"]

# ── Title ─────────────────────────────────────────
st.title("💳 EMI Eligibility Checker")
st.write("Check your EMI eligibility instantly using AI")

# ═══════════════════════════════════════════════════
# Income
# ═══════════════════════════════════════════════════

st.header("Income & Employment")

col1, col2 = st.columns(2)

with col1:
    monthly_salary = st.number_input("Monthly Salary", 0.0, value=50000.0)
    years_of_employment = st.number_input("Years of Employment", 0.0, value=3.0)

with col2:
    bank_balance = st.number_input("Bank Balance", 0.0, value=100000.0)
    emergency_fund = st.number_input("Emergency Fund", 0.0, value=30000.0)

# ═══════════════════════════════════════════════════
# Expenses
# ═══════════════════════════════════════════════════

st.header("Expenses")

col1, col2 = st.columns(2)

with col1:
    monthly_rent = st.number_input("Monthly Rent", 0.0, value=12000.0)
    other_expenses = st.number_input("Other Expenses", 0.0, value=8000.0)

with col2:
    existing_loans = st.number_input("Existing Loans", 0, value=1)
    current_emi_amount = st.number_input("Current EMI Amount", 0.0, value=5000.0)

# ═══════════════════════════════════════════════════
# Loan Request
# ═══════════════════════════════════════════════════

st.header("Loan Request")

col1, col2, col3 = st.columns(3)

with col1:
    requested_amount = st.number_input("Requested Loan Amount", 0.0, value=200000.0)

with col2:
    requested_tenure = st.number_input("Requested Tenure (months)", 1, value=24)

with col3:
    credit_score = st.number_input("Credit Score", 300, 900, value=700)

# ═══════════════════════════════════════════════════
# Calculations
# ═══════════════════════════════════════════════════

total_expenses = monthly_rent + other_expenses + current_emi_amount

if monthly_salary > 0:
    savings_ratio = (monthly_salary - total_expenses) / monthly_salary
    debt_to_income = current_emi_amount / monthly_salary
    loan_to_income_ratio = requested_amount / (monthly_salary * 12)
    expense_to_income = total_expenses / monthly_salary
else:
    savings_ratio = 0
    debt_to_income = 0
    loan_to_income_ratio = 0
    expense_to_income = 0

st.subheader("Financial Snapshot")

st.write("Savings Ratio:", round(savings_ratio, 2))
st.write("Debt to Income:", round(debt_to_income, 2))
st.write("Expense to Income:", round(expense_to_income, 2))

# ═══════════════════════════════════════════════════
# Prediction
# ═══════════════════════════════════════════════════

if st.button("Check Eligibility"):

    clf_features = np.array([[

        monthly_salary,
        bank_balance,
        emergency_fund,
        savings_ratio,
        existing_loans,
        current_emi_amount,
        debt_to_income,
        loan_to_income_ratio,
        expense_to_income,
        requested_amount,
        requested_tenure,
        years_of_employment,
        monthly_rent,
        total_expenses,
        credit_score

    ]])

    prediction = classifier.predict(clf_features)[0]

    if prediction == 1:

        st.success("✅ You are eligible for EMI")

        reg_features = np.array([[

            existing_loans,
            1,
            monthly_salary,
            total_expenses,
            expense_to_income,
            credit_score,
            bank_balance,
            debt_to_income,
            1,
            savings_ratio

        ]])

        max_emi = regressor.predict(reg_features)[0]

        st.subheader("Maximum EMI You Can Afford")
        st.write(f"₹ {int(max_emi):,}")

    else:

        st.error("❌ You are not eligible for EMI currently")

st.write("---")
st.caption("AI powered EMI eligibility system")