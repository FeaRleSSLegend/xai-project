import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

FEATURE_ORDER = [
    "price",
    "sentiment_score",
    "review_length",
    "review_votes",
    "brand_avg_rating",
    "brand_review_count",
    "price_tier",
]

PRICE_TIERS = {
    "Budget": 0,
    "Mid-range": 1,
    "Premium": 2,
    "Ultra-premium": 3,
}


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/recommendation_model.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    brand_lookup = joblib.load("models/brand_lookup.pkl")
    analyzer = SentimentIntensityAnalyzer()
    return model, explainer, brand_lookup, analyzer


def main():
    st.set_page_config(page_title="Product Recommendation", layout="centered")
    st.title("Product Recommendation")

    model, explainer, brand_lookup, analyzer = load_artifacts()

    col1, col2 = st.columns(2)
    with col1:
        price = st.number_input("Price", min_value=0.0, max_value=2000.0, value=50.0, step=1.0)
        price_tier_label = st.selectbox("Price tier", list(PRICE_TIERS.keys()))
    with col2:
        brand_name = st.selectbox("Brand", list(brand_lookup.index))

    review_text = st.text_area("Review text", height=140, placeholder="Type the customer review here...")

    if st.button("Predict", type="primary"):
        sentiment_score = analyzer.polarity_scores(review_text or "")["compound"]
        review_length = len(review_text or "")
        review_votes = 0
        price_tier = PRICE_TIERS[price_tier_label]
        brand_avg_rating = float(brand_lookup.loc[brand_name, "brand_avg_rating"])
        brand_review_count = float(brand_lookup.loc[brand_name, "brand_review_count"])

        row = {
            "price": price,
            "sentiment_score": sentiment_score,
            "review_length": review_length,
            "review_votes": review_votes,
            "brand_avg_rating": brand_avg_rating,
            "brand_review_count": brand_review_count,
            "price_tier": price_tier,
        }
        X = pd.DataFrame([[row[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)

        proba = model.predict_proba(X)[0, 1]
        st.markdown(f"## {proba * 100:.1f}%")
        st.caption("Recommendation likelihood")

        shap_values = explainer.shap_values(X)
        expected_value = explainer.expected_value
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            expected_value = expected_value[1] if hasattr(expected_value, "__len__") else expected_value

        st.subheader("Explanation")
        shap.force_plot(
            expected_value,
            shap_values[0, :] if shap_values.ndim > 1 else shap_values,
            X.iloc[0, :],
            matplotlib=True,
            show=False,
        )
        fig = plt.gcf()
        st.pyplot(fig, bbox_inches="tight")
        plt.close(fig)

        if proba >= 0.7:
            msg = "Strong signal that this product will be recommended."
        elif proba >= 0.5:
            msg = "Leaning toward a recommendation, but the signal is moderate."
        elif proba >= 0.3:
            msg = "Unlikely to be recommended — a few features pull against it."
        else:
            msg = "Low likelihood of recommendation."
        st.write(msg)


if __name__ == "__main__":
    main()
