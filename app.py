import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from groq import Groq
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

GROQ_SYSTEM_PROMPT = (
    "You are explaining an AI product recommendation system to a non-technical "
    "user. Given SHAP feature contributions, explain in exactly 3 plain sentences: "
    "what drove this recommendation score, which features helped and which hurt, "
    "and what this means for the product overall. Be specific about feature names "
    "and values. No bullet points, just flowing sentences."
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/recommendation_model.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    brand_lookup = joblib.load("models/brand_lookup.pkl")
    X_train_sample = np.load("models/X_train_sample.npy")
    lime_explainer = LimeTabularExplainer(
        training_data=X_train_sample,
        feature_names=FEATURE_ORDER,
        class_names=["Not Recommended", "Recommended"],
        mode="classification",
    )
    analyzer = SentimentIntensityAnalyzer()
    return model, explainer, brand_lookup, lime_explainer, analyzer


def build_shap_summary(shap_values_row, feature_names, proba):
    lines = ["Feature contributions to this recommendation prediction:"]
    for name, val in zip(feature_names, shap_values_row):
        lines.append(
            f"{name}: {val:+.2f} (positive = pushes toward recommended, "
            f"negative = pushes against)"
        )
    lines.append(f"Overall prediction: {proba * 100:.1f}% recommendation likelihood")
    return "\n".join(lines)


def groq_explanation(summary_text):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": GROQ_SYSTEM_PROMPT},
            {"role": "user", "content": summary_text},
        ],
    )
    return resp.choices[0].message.content


def main():
    st.set_page_config(page_title="Product Recommendation", layout="centered")
    st.title("Product Recommendation")

    model, explainer, brand_lookup, lime_explainer, analyzer = load_artifacts()

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
        x_array = X.to_numpy()[0]

        proba = model.predict_proba(X)[0, 1]
        st.markdown(f"## {proba * 100:.1f}%")
        st.caption("Recommendation likelihood")

        shap_values = explainer.shap_values(X)
        expected_value = explainer.expected_value
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            expected_value = expected_value[1] if hasattr(expected_value, "__len__") else expected_value
        shap_row = shap_values[0, :] if shap_values.ndim > 1 else shap_values

        st.subheader("SHAP Explanation")
        shap.force_plot(
            expected_value,
            shap_row,
            X.iloc[0, :],
            matplotlib=True,
            show=False,
        )
        fig = plt.gcf()
        st.pyplot(fig, bbox_inches="tight")
        plt.close(fig)

        st.subheader("LIME Explanation")
        lime_exp = lime_explainer.explain_instance(
            data_row=x_array,
            predict_fn=lambda arr: model.predict_proba(
                pd.DataFrame(arr, columns=FEATURE_ORDER)
            ),
            num_features=len(FEATURE_ORDER),
        )
        lime_fig = lime_exp.as_pyplot_figure()
        st.pyplot(lime_fig, bbox_inches="tight")
        plt.close(lime_fig)

        st.subheader("Plain-language Explanation")
        summary_text = build_shap_summary(shap_row, FEATURE_ORDER, proba)
        try:
            st.write(groq_explanation(summary_text))
        except Exception as e:
            st.error(f"Could not generate Groq explanation: {e}")


if __name__ == "__main__":
    main()
