import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)

model = joblib.load("fake_news_detector_model.joblib")


st.markdown("""
<h1 style='text-align:center;'>📰 Fake News Detection System</h1>
<p style='text-align:center; color: gray; font-size:16px;'>
Analyze news text and determine whether it is <b>Fake</b> or <b>True</b> using a trained Machine Learning model.
</p>
""", unsafe_allow_html=True)

st.markdown("---")


st.markdown("### 📝 News Text Input")

news_text = st.text_area(
    "",
    height=200,
    placeholder="Paste the news article here..."
)


if st.button("🔍 Analyze"):
    if news_text.strip() == "":
        st.warning("Please enter a news text.")
    else:
        prediction = model.predict([news_text])[0]
        probabilities = model.predict_proba([news_text])[0]

        fake_prob = probabilities[0]
        real_prob = probabilities[1]

        st.markdown("---")


        if prediction == 0:
            st.markdown("""
            <div style='background-color:#ffe6e6; padding:20px; border-radius:12px;'>
                <h3 style='color:#cc0000;'>❌ Prediction: FAKE NEWS</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color:#e6ffe6; padding:20px; border-radius:12px;'>
                <h3 style='color:#006600;'>✅ Prediction: REAL NEWS</h3>
            </div>
            """, unsafe_allow_html=True)


        st.markdown("### 📊 Prediction Confidence")

        st.write("**Fake Probability**")
        st.progress(fake_prob)
        st.write(f"P(FAKE): {fake_prob:.2f}")

        st.write("**Real Probability**")
        st.progress(real_prob)
        st.write(f"P(REAL): {real_prob:.2f}")


        st.markdown("###  Probability Visualization")

        labels = ["Fake", "Real"]
        values = [fake_prob, real_prob]
        colors = ["#cc0000", "#006600"]  # Red / Green

        fig, ax = plt.subplots()
        ax.bar(labels, values, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence Comparison")

        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

        st.pyplot(fig)

st.markdown("---")

st.markdown("""
<div style='background-color:#f0f2f6; padding:12px; border-radius:10px; text-align:center;'>
<b>Model Accuracy:</b> 98.5%
</div>
""", unsafe_allow_html=True)

st.caption("⚠️ This interface is a demonstration UI developed for an academic Machine Learning project.")
