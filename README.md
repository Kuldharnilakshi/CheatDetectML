# Online Exam Cheating Detection System

## Overview
This is an interactive ML web application built using **Streamlit** that predicts the probability of a student cheating during online exams. Users can input exam behavior data and get a prediction along with a feature importance chart. The app also allows generating a downloadable report with input details and prediction results.

## Features
- Predicts **cheating probability** using a trained Random Forest model
- Displays **prediction result** with probability
- Shows **Feature Importance** to explain model decisions
- Clean, responsive, and interactive **Streamlit UI**

## Inputs
The app takes the following features:
- `Tab Switch Count` – Number of times student switched browser tabs
- `Average Answer Time` – Time taken to answer each question
- `Question Revisit Count` – Number of times questions were revisited
- `Idle Time` – Time spent idle during the exam
- `Copy-Paste Events` – Number of copy-paste actions

## Tech Stack
- **Python** – Core language
- **Scikit-learn** – Machine Learning (Random Forest)
- **Pandas** – Data handling
- **Matplotlib** – Visualization
- **SHAP** – Explainable ML / Feature Importance
- **Streamlit** – Web UI and deployment

