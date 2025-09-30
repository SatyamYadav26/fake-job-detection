import streamlit as st
import joblib
import pandas as pd

# Load saved model and vectorizer
model = joblib.load("fake_job_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Streamlit UI
st.title("🕵️ Fake Job Posting Detector")
st.write("Paste any job description below and the model will predict if it's **REAL** or **FAKE** with confidence score.")

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []  
# User input
user_input = st.text_area("Paste a job description:")

if st.button("Check Job"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a job description.")
    else:
        # Transform input text
        features = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        prob_real = round(probability[0] * 100, 2)
        prob_fake = round(probability[1] * 100, 2)

        # Show result
        if prediction == 1:
            result_text = f"🚨 FAKE Job (Confidence: {prob_fake}% Fake | {prob_real}% Real)"
            st.error(result_text)
        else:
            result_text = f"✅ REAL Job (Confidence: {prob_real}% Real | {prob_fake}% Fake)"
            st.success(result_text)

        # Save to history (limit to last 5 entries)
        st.session_state.history.insert(0, (user_input, prediction, prob_real, prob_fake))
        st.session_state.history = st.session_state.history[:5]

# Show history
if st.session_state.history:
    st.subheader("📜 Last 5 Predictions")
    for i, (text, prediction, prob_real, prob_fake) in enumerate(st.session_state.history, 1):
        with st.expander(f"Example {i}:"):
            st.write(f"**Job Posting:** {text}")
            if prediction == 1:
                st.write(f"**Prediction:** 🚨 FAKE Job ({prob_fake}% Fake | {prob_real}% Real)")
            else:
                st.write(f"**Prediction:** ✅ REAL Job ({prob_real}% Real | {prob_fake}% Fake)")

    # Convert history to DataFrame
    df_history = pd.DataFrame(st.session_state.history, columns=["Job Description", "Prediction (0=Real,1=Fake)", "Prob_Real (%)", "Prob_Fake (%)"])

    # Download button
    st.download_button(
        label="📥 Download History as CSV",
        data=df_history.to_csv(index=False).encode("utf-8"),
        file_name="job_predictions_history.csv",
        mime="text/csv"
    )
