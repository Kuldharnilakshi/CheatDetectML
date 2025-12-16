import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io

# Load trained model
model = joblib.load("model.pkl")

# Web page title
st.title("Online Exam Cheating Detection")
st.write("Enter the student’s exam behavior details below:")

# User inputs
tab_switch_count = st.number_input("Tab Switch Count", min_value=0, max_value=20, value=0)
avg_answer_time = st.number_input("Average Answer Time (seconds)", min_value=1, max_value=60, value=25)
question_revisit_count = st.number_input("Question Revisit Count", min_value=0, max_value=20, value=2)
idle_time = st.number_input("Idle Time (seconds)", min_value=0, max_value=600, value=50)
copy_paste_events = st.number_input("Copy-Paste Events", min_value=0, max_value=5, value=0)

# Prepare input dataframe
input_data = pd.DataFrame({
    "tab_switch_count": [tab_switch_count],
    "avg_answer_time": [avg_answer_time],
    "question_revisit_count": [question_revisit_count],
    "idle_time": [idle_time],
    "copy_paste_events": [copy_paste_events]
})

# Predict button
if st.button("Predict Cheating"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]  # Probability of cheating
    
    if prediction == 1:
        st.error(f"🔴 Cheating Detected! Probability: {proba*100:.2f}%")
    else:
        st.success(f"🟢 No Cheating Detected. Probability: {proba*100:.2f}%")

# Feature Importance button
if st.button("Show Feature Importance"):
    st.subheader("📊 Feature Importance")
    importance = model.feature_importances_
    features = input_data.columns

    # Sort features by importance
    sorted_idx = importance.argsort()
    sorted_features = features[sorted_idx]
    sorted_importance = importance[sorted_idx]

    # Create bar chart with UI-friendly colors
    fig, ax = plt.subplots(figsize=(6,4))
    colors = plt.cm.Blues(sorted_importance / max(sorted_importance))  # Gradient blue
    ax.barh(sorted_features, sorted_importance, color=colors)

    # Minimalist style
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance in Cheating Prediction", pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color("#DDDDDD")
    ax.tick_params(axis='x', colors='#555555')
    ax.tick_params(axis='y', colors='#555555')

    # Add value labels
    for i, val in enumerate(sorted_importance):
        ax.text(val + 0.01, i, f"{val:.2f}", va='center', ha='left', color='#333333', fontsize=10)

    st.pyplot(fig, use_container_width=True)

    