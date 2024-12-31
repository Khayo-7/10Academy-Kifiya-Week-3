# Filename: app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.visualization import plot_savings, plot_risk_distribution

# Initialize Streamlit Page
st.set_page_config(page_title="AlphaCare Risk Analysis Dashboard", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Overview", "EDA Highlights", "Hypothesis Testing", "Recommendations"])

# Load Dataset Summary (Placeholder Function)
@st.cache_data
def load_data_summary():
    return pd.read_csv("data/dataset_summary.csv")

if section == "Overview":
    st.title("AlphaCare Risk Analysis Dashboard")
    st.subheader("Introduction")
    st.write("""
        This dashboard showcases interim findings from the AlphaCare Risk Analysis Challenge, 
        covering exploratory data analysis, hypothesis testing, and actionable insights.
    """)

    # Dataset Stats
    st.markdown("### Dataset Overview")
    dataset_summary = load_data_summary()
    st.write(dataset_summary)

elif section == "EDA Highlights":
    st.title("Exploratory Data Analysis Highlights")
    st.markdown("### Missing Data Analysis")
    st.image("screenshots/image.png", caption="Missing Data Heatmap")

    st.markdown("### Correlation Analysis")
    st.image("screenshots/image-5.png", caption="Key Correlations")

elif section == "Hypothesis Testing":
    st.title("Hypothesis Testing Results")
    st.subheader("Test Outcomes")
    test_results = pd.DataFrame({
        "Hypothesis": [
            "Risk differences across provinces",
            "Risk differences between zip codes",
            "Margin differences across zip codes",
            "Risk differences between genders"
        ],
        "Test Method": ["ANOVA", "T-test", "ANOVA", "T-test"],
        "p-value": [3.33e-06, 4.80e-11, 0.999, 0.635],
        "Significant?": ["Yes", "Yes", "No", "No"]
    })
    st.write(test_results)

elif section == "Recommendations":
    st.title("Recommendations")
    st.subheader("Key Actions")
    st.write("""
        1. Optimize premium structures using insights from geographical risk variations.
        2. Address missing data and anomalies to enhance data quality.
        3. Implement region-specific pricing strategies for optimal profitability.
    """)

    # Plot Savings
    st.markdown("### Savings by Location")
    savings_data = pd.DataFrame({
        "Location": ["Province A", "Province B", "Zip 001", "Zip 005"],
        "Savings %": [25, 18, 22, 19]
    })
    fig, ax = plt.subplots()
    plot_savings(savings_data, "Location", "Savings %", "Savings Opportunities")
    st.pyplot(fig)
