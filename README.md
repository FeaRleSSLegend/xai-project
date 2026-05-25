# Product Recommendation System

A machine learning system that predicts how likely a product is to be 
recommended, with full explainability using SHAP and LIME.

Built on the Amazon Unlocked Mobile Phones dataset (400k reviews).

## What it does

- Predicts recommendation likelihood (0-100%) for a product
- Explains why using SHAP force plots
- Identifies bias across price groups, brand size, and sentiment

## Features used

- Review sentiment (extracted automatically from review text via VADER)
- Review length (extracted automatically from review text)
- Price and price tier
- Brand average rating and review count

## Stack

- XGBoost — classification model
- SHAP — global and local explainability
- LIME — local explainability
- VADER — sentiment analysis
- Streamlit — deployed interface

## Run locally

pip install -r requirements.txt
streamlit run app.py

## Dataset

Amazon Unlocked Mobile Phones — via Kaggle
https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones

## Links

- **Live App**: https://xai-project-g7.streamlit.app/
- **Demo Video**: https://youtu.be/W6WqgXfzifs