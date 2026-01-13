import streamlit as st
import pandas as pd
from core.data_loader import load_data
from core.schema_analyzer import analyze_schema
from core.analysis_engine import statistical_summary
from core.insights import generate_insights
from core.visualizer import plot_distribution, plot_category_counts
from core.ml_engine import train_model

st.set_page_config(page_title="Data Analyst System", layout="wide")

st.title("📊 Automated Data Analyst System")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    schema = analyze_schema(df)

    st.subheader("🧠 Dataset Schema")
    st.json(schema)

    summary = statistical_summary(df)

    st.subheader("📈 Statistical Summary")
    st.dataframe(summary)

    insights = generate_insights(schema, summary)

    st.subheader("💡 Key Insights")
    for insight in insights:
        st.write("•", insight)

    st.subheader("📊 Visual Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if schema["numeric_columns"]:
            num_col = st.selectbox(
                "Select numerical column",
                schema["numeric_columns"]
            )
            plot_distribution(df, num_col)

    with col2:
        if schema["categorical_columns"]:
            cat_col = st.selectbox(
                "Select categorical column",
                schema["categorical_columns"]
            )
            plot_category_counts(df, cat_col)
    
    st.subheader("🤖 Machine Learning")

    target_col = st.selectbox("Select Target Column", df.columns)

    if st.button("Train Model"):
        with st.spinner("Running AutoML..."):
            results = train_model(df, target_col)

        st.success(f"Best Model: {results['model']}")

        if results["type"] == "regression":
            st.metric("RMSE", round(results["rmse"], 3))
            st.metric("R² Score", round(results["r2"], 3))
            st.metric("CV RMSE", round(results["cv_rmse"], 3))

        else:
            st.metric("Accuracy", round(results["accuracy"], 3))
            st.metric("Precision", round(results["precision"], 3))
            st.metric("Recall", round(results["recall"], 3))
            st.metric("F1 Score", round(results["f1"], 3))
            st.metric("CV Accuracy", round(results["cv_accuracy"], 3))

        if results["feature_importance"] is not None:
            st.subheader("📌 Feature Importance")

            imp_df = pd.DataFrame({
                "Feature": results["features"],
                "Importance": results["feature_importance"]
            }).sort_values(by="Importance", ascending=False)

            st.dataframe(imp_df.head(10))


