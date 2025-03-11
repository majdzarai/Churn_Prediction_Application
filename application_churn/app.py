import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, ConfusionMatrixDisplay

# Load the saved model and scaler
model = joblib.load("gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")

# Adjusted threshold
optimal_threshold = -0.476

# Streamlit App Title
st.title("✨ Churn Prediction App 🚀")

# Sidebar for Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Choose a page", ["📖 How to Use the Model", "🔍 Start Predicting", "📈 Statistics & Metrics"])

# ---------------------------- #
# PAGE 1: How to Use the Model #
# ---------------------------- #
if page == "📖 How to Use the Model":
    st.header("📖 How to Use the Churn Prediction App")
    st.write("""
    ### What is This App For?  
    This app helps to **predict customer churn** based on input data like account length, total call minutes, and other attributes.  
    It uses a **Gradient Boosting Model** trained with historical churn data.
    """)

# ----------------------- #
# PAGE 2: Start Predicting #
# ----------------------- #
elif page == "🔍 Start Predicting":
    st.header("🔍 Start Predicting Churn")
    st.write("Enter customer data to predict churn probability.")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            account_length = st.number_input("Account Length", min_value=0, value=100)
            area_code = st.number_input("Area Code", min_value=0, value=415)
            international_plan = st.radio("International Plan", ["No", "Yes"], horizontal=True)
            voice_mail_plan = st.radio("Voice Mail Plan", ["No", "Yes"], horizontal=True)
            customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=1)
        
        with col2:
            total_day_minutes = st.slider("Total Day Minutes", min_value=0.0, max_value=500.0, value=100.0)
            total_eve_minutes = st.slider("Total Eve Minutes", min_value=0.0, max_value=500.0, value=150.0)
            total_night_minutes = st.slider("Total Night Minutes", min_value=0.0, max_value=500.0, value=200.0)
            total_intl_minutes = st.slider("Total Intl Minutes", min_value=0.0, max_value=50.0, value=10.0)
    
    input_data = pd.DataFrame({
        'Account length': [account_length],
        'Area code': [area_code],
        'International plan': [1 if international_plan == "Yes" else 0],
        'Voice mail plan': [1 if voice_mail_plan == "Yes" else 0],
        'Total day minutes': [total_day_minutes],
        'Total eve minutes': [total_eve_minutes],
        'Total night minutes': [total_night_minutes],
        'Total intl minutes': [total_intl_minutes],
        'Customer service calls': [customer_service_calls]
    })

    if st.button("✨ Predict Churn"):
        with st.spinner("🔍 Predicting... Please wait"):
            time.sleep(2)
            scaled_input = scaler.transform(input_data)
            y_score = model.decision_function(scaled_input)
            y_pred = (y_score > optimal_threshold).astype(int)

            st.subheader("🎯 Prediction Result:")
            if y_pred[0] == 1:
                st.success("The customer is likely to **Churn** 💔")
            else:
                st.success("The customer is **Not Likely to Churn** ✅")

            st.session_state['latest_score'] = y_score[0]

# ----------------------------- #
# PAGE 3: Statistics & Metrics  #
# ----------------------------- #
elif page == "📈 Statistics & Metrics":
    st.header("📈 Statistics & Model Performance")

    if 'latest_score' not in st.session_state:
        st.warning("⚠️ No predictions made yet! Go to 'Start Predicting' to make a prediction first.")
    else:
        st.write("### Last Prediction Raw Score:")
        st.write(f"**{st.session_state['latest_score']:.3f}**")

        y_true = [0, 1, 0, 1]  # Placeholder true values
        y_pred = [0, 1, 1, 1]  # Placeholder predicted values
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        st.write("### Confusion Matrix:")
        fig, ax = plt.subplots()
        disp.plot(cmap="Blues", ax=ax)
        st.pyplot(fig)

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        st.write("### ROC Curve:")
        fig, ax = plt.subplots()
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend()
        st.pyplot(fig)