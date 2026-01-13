import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_distribution(df, column):
    fig, ax = plt.subplots()
    sns.histplot(df[column].dropna(), kde=True, ax=ax)
    st.pyplot(fig)


def plot_category_counts(df, column):
    fig, ax = plt.subplots()
    df[column].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)
