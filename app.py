import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline (preprocess + model)
model = joblib.load("xgb_model.pkl")

st.title("Reddit Popularity Predictor")

st.write("Enter Reddit post features:")

# --- INPUTS ---
subreddit = st.text_input("Subreddit", "AskReddit")
post_hour = st.slider("Post Hour", 0, 23, 12)
post_day = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
post_month = st.slider("Month", 1, 12, 6)

num_comments = st.number_input("Number of Comments", 0, 10000, 50)
title_len = st.number_input("Title Length", 0, 500, 50)
body_len = st.number_input("Body Length", 0, 5000, 200)

flair = st.text_input("Flair", "None")

# --- PREDICTION ---
if st.button("Predict Popularity"):

    input_data = pd.DataFrame([{
        "subreddit": subreddit,
        "post_hour": post_hour,
        "post_day": post_day,
        "post_month": post_month,
        "num_comments": num_comments,
        "title_length": title_len,
        "body_length": body_len,
        "flair": flair
    }])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if pred == 1:
        st.success(f"Popular (Confidence: {prob:.3f})")
    else:
        st.error(f"Not Popular (Confidence: {prob:.3f})")