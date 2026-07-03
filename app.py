import streamlit as st
import pandas as pd
import joblib



# ==========================
# Load Trained Model
# ==========================
model = joblib.load("model/loan_model.pkl")
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Intelligent Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# ==========================
# Header
# ==========================
st.markdown("""
# 🏦 Intelligent Loan Approval Prediction

### AI Powered Banking Decision Support System
""")

st.caption("Predict whether a loan application is likely to be approved using Machine Learning.")

st.write("---")

# ==========================
# Sidebar
# ==========================
st.sidebar.title("🏦 Loan Prediction System")

st.sidebar.markdown("---")

st.sidebar.success("✅ Machine Learning Project")

st.sidebar.markdown("""
### 📌 Project Features

- 🤖 AI Loan Prediction
- 📊 Real-time Decision
- 💰 Financial Analysis
- 🏠 Property Evaluation
- 📈 Instant Result
""")

st.sidebar.markdown("---")

st.sidebar.write("👩‍💻 Developed by Anika")

# ==========================
# Two Columns
# ==========================
col1, col2 = st.columns(2)

with col1:

    st.subheader("👤 Applicant Details")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Marital Status",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Employment",
        ["No", "Yes"]
    )

with col2:

    st.subheader("💰 Financial Details")

    applicant_income = st.number_input(
        "Applicant Income (₹)",
        min_value=0
    )

    coapplicant_income = st.number_input(
        "Co-Applicant Income (₹)",
        min_value=0
    )

    loan_amount = st.number_input(
        "Loan Amount (Thousands ₹)",
        min_value=0
    )

    loan_term = st.number_input(
        "Loan Term (Months)",
        value=360
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0]
    )

st.write("---")

st.subheader("🏠 Property Details")

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

st.write("")

# ==========================
# Predict Button
# ==========================
c1, c2, c3 = st.columns([1,2,1])

with c2:
    predict_btn = st.button(
        "🏦 Predict Loan Approval",
        use_container_width=True
    )

st.write("---")

st.subheader("📊 Prediction Result")

# ==========================
# Prediction
# ==========================
if predict_btn:

    input_data = pd.DataFrame([{

        "Gender": 1 if gender == "Male" else 0,

        "Married": 1 if married == "Yes" else 0,

        "Dependents": {
            "0": 0,
            "1": 1,
            "2": 2,
            "3+": 3
        }[dependents],

        "Education": 0 if education == "Graduate" else 1,

        "Self_Employed": 1 if self_employed == "Yes" else 0,

        "ApplicantIncome": applicant_income,

        "CoapplicantIncome": coapplicant_income,

        "LoanAmount": loan_amount,

        "Loan_Amount_Term": loan_term,

        "Credit_History": credit_history,

        "Property_Area": {
            "Rural": 0,
            "Semiurban": 1,
            "Urban": 2
        }[property_area]

    }])

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    if prediction[0] == 1:

        st.success("🎉 Congratulations! Loan Approved")

        st.progress(int(probability[0][1] * 100))

        st.metric(
            "Approval Probability",
            f"{probability[0][1]*100:.2f}%"
        )

    else:

        st.error("❌ Sorry! Loan Rejected")

        st.progress(int(probability[0][0] * 100))

        st.metric(
            "Rejection Probability",
            f"{probability[0][0]*100:.2f}%"
        )