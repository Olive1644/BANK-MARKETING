import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- THEME 
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: #1E88E5;
}

.stDataFrame {
    border-radius: 10px;
}

.stSelectbox {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.subheader("📊 Model Performance Comparison")

# ---------------- DATA ----------------
data = {
    "Model": ["Random Forest", "XGBoost"],
    "Accuracy": [0.91, 0.9167],
    "F1 Score": [0.5485, 0.5925],
    "Log Loss": [0.1769, 0.1743]
}

df = pd.DataFrame(data)

# ---------------- TABLE ----------------
st.dataframe(df, use_container_width=True)

# ---------------- BAR CHART 1 ----------------
fig, ax = plt.subplots()
df.set_index("Model").plot(kind="bar", ax=ax)

ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")

st.pyplot(fig)

# ---------------- METRIC SELECT ----------------
metric = st.selectbox("Select Metric", ["Accuracy", "F1 Score", "Log Loss"])

# ---------------- BAR CHART 2 ----------------
fig, ax = plt.subplots()

ax.bar(df["Model"], df[metric])

ax.set_title(f"{metric} Comparison")
ax.set_ylabel(metric)

st.pyplot(fig)