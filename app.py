import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import xgboost

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="Lead Scoring", page_icon="🎯", layout="centered")

# ============================================================
# Load model
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("lead_model.pkl")

pipe = load_model()

# ============================================================
# All raw columns the pipeline was fit on (order matters for some setups)
# Adjust this list if your pipeline expects different columns.
# ============================================================
ALL_COLUMNS = [
    "Lead Origin", "Lead Source", "TotalVisits",
    "Total Time Spent on Website", "Last Activity",
    "What is your current occupation",
    "Do Not Email", "Specialization", "Lead Profile",
    "incomplete_profile", "asym_info_available", "Page Views Per Visit",
    "A free copy of Mastering The Interview"
    # add any other columns your pipeline was fit on
]

# ============================================================
# Header
# ============================================================
st.title("🎯 Lead Scoring — X Education")
st.write("Enter the lead details below to predict whether the lead will convert.")

# ============================================================
# Inputs
# ============================================================
col1, col2 = st.columns(2)

with col1:
    lead_origin = st.selectbox(
        "Lead Origin",
        ["Landing Page Submission", "API", "Lead Add Form", "Lead Import"],
    )
    lead_source = st.selectbox(
        "Lead Source",
        ["Google", "Direct Traffic", "Olark Chat", "Organic Search",
         "Reference", "Facebook", "Welingak Website"],
    )
    total_visits = st.number_input("Total Visits", min_value=0, value=5, step=1)

with col2:
    total_time = st.number_input(
        "Total Time Spent on Website (seconds)", min_value=0, value=500, step=10
    )
    last_activity = st.selectbox(
        "Last Activity",
        ["Email Opened", "SMS Sent", "Olark Chat Conversation",
         "Page Visited on Website", "Email Bounced", "Unsubscribed"],
    )
    occupation = st.selectbox(
        "What is your current occupation",
        ["Working Professional", "Student", "Unemployed", "Other"],
    )

# ============================================================
# Predict
# ============================================================
if st.button("🎯 Predict Lead", type="primary", use_container_width=True):
    # Build a full row: all columns = NaN, then overwrite the 6 user inputs
    row = {col: np.nan for col in ALL_COLUMNS}
    row["Lead Origin"] = lead_origin
    row["Lead Source"] = lead_source
    row["TotalVisits"] = total_visits
    row["Total Time Spent on Website"] = total_time
    row["Last Activity"] = last_activity
    row["What is your current occupation"] = occupation

    input_df = pd.DataFrame([row])

    try:
        prob = pipe.predict_proba(input_df)[0, 1]
        score = int(prob * 100)

        st.divider()
        st.metric("Conversion Probability", f"{score} / 100")

        if prob >= 0.5:
            st.markdown(
                f"<div style='padding:16px;border-radius:8px;"
                f"background-color:#d4edda;color:#155724;"
                f"font-size:20px;font-weight:600;text-align:center;'>"
                f"🔥 HOT LEAD — likely to convert</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='padding:16px;border-radius:8px;"
                f"background-color:#f8d7da;color:#721c24;"
                f"font-size:20px;font-weight:600;text-align:center;'>"
                f"❄️ COLD LEAD — unlikely to convert</div>",
                unsafe_allow_html=True,
            )

        with st.expander("🔍 See input details"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that ALL_COLUMNS in app.py matches the columns your "
            "pipeline was trained on."
        )