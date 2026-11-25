import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="House Prices EDA", page_icon="🏠", layout="wide")

st.title("🏠 House Prices – Exploratory Data Analysis")
st.markdown("Interactive analysis of the Ames Housing Dataset.")

uploaded_file = st.file_uploader("Upload your train.csv dataset", type=["csv"])

if uploaded_file is None:
    st.info("Please upload the Kaggle House Prices train.csv dataset to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

if "SalePrice" not in df.columns:
    st.error("This CSV does not contain the required SalePrice column.")
    st.stop()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose Analysis", [
    "Overview", "Dataset Explorer", "Statistics",
    "Distributions", "Correlation", "Outliers", "Dashboard"
])

if page == "Overview":
    st.header("Dataset Overview")
    a, b, c = st.columns(3)
    a.metric("Total Houses", f"{len(df):,}")
    b.metric("Total Features", len(df.columns))
    c.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Data Types")
    st.dataframe(df.dtypes.astype(str).rename("Data Type"))

elif page == "Dataset Explorer":
    st.header("Dataset Explorer")
    st.write("Dataset shape:", df.shape)
    st.dataframe(df, use_container_width=True)
    st.subheader("Missing Values")
    missing = df.isnull().sum().sort_values(ascending=False)
    st.dataframe(missing[missing > 0].rename("Missing Count"))

elif page == "Statistics":
    st.header("Descriptive Statistics")
    st.dataframe(df.describe().T, use_container_width=True)
    st.subheader("SalePrice Statistics")
    st.write(df["SalePrice"].describe())

elif page == "Distributions":
    st.header("Distribution Analysis")
    column = st.selectbox("Choose numerical feature", df.select_dtypes(include=np.number).columns)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[column].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")
    st.pyplot(fig)
    if "OverallQual" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df, x="OverallQual", y="SalePrice", ax=ax)
        ax.set_title("SalePrice by Overall Quality")
        st.pyplot(fig)

elif page == "Correlation":
    st.header("Correlation Analysis")
    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)
    if "SalePrice" in corr:
        st.subheader("Top Features Correlated with SalePrice")
        st.dataframe(corr["SalePrice"].drop("SalePrice").sort_values(ascending=False).head(10))

elif page == "Outliers":
    st.header("Outlier Analysis")
    column = st.selectbox("Select feature", ["SalePrice", "GrLivArea", "LotArea"])
    if column in df.columns:
        values = df[column].dropna()
        q1, q3 = values.quantile(.25), values.quantile(.75)
        iqr = q3 - q1
        outliers = values[(values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)]
        st.metric("Potential IQR Outliers", len(outliers))
        fig, ax = plt.subplots()
        sns.boxplot(x=values, ax=ax)
        st.pyplot(fig)

elif page == "Dashboard":
    st.header("House Prices EDA Dashboard")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    if "YearBuilt" in df.columns:
        yearly = df.groupby("YearBuilt")["SalePrice"].mean()
        axes[0, 0].plot(yearly.index, yearly.values)
        axes[0, 0].set_title("Average SalePrice by Year Built")
    else:
        axes[0, 0].text(.5, .5, "YearBuilt unavailable", ha="center")

    if "Neighborhood" in df.columns:
        neighborhood = df.groupby("Neighborhood")["SalePrice"].mean().nlargest(10)
        axes[0, 1].barh(neighborhood.index, neighborhood.values)
        axes[0, 1].set_title("Top 10 Neighborhoods")
        axes[0, 1].invert_yaxis()
    else:
        axes[0, 1].text(.5, .5, "Neighborhood unavailable", ha="center")

    sns.histplot(df["SalePrice"], bins=30, kde=True, ax=axes[1, 0])
    axes[1, 0].axvline(df["SalePrice"].mean(), color="red", linestyle="--", label="Mean")
    axes[1, 0].axvline(df["SalePrice"].median(), color="green", linestyle="--", label="Median")
    axes[1, 0].set_title("SalePrice Distribution")
    axes[1, 0].legend()

    if "GrLivArea" in df.columns and "OverallQual" in df.columns:
        sns.scatterplot(data=df, x="GrLivArea", y="SalePrice", hue="OverallQual",
                        palette="viridis", ax=axes[1, 1], legend=False)
        axes[1, 1].set_title("Living Area vs SalePrice by Quality")
    else:
        axes[1, 1].text(.5, .5, "Required features unavailable", ha="center")

    fig.tight_layout()
    st.pyplot(fig)

st.sidebar.success("House Prices EDA")
st.caption("Built with Python, Pandas, Matplotlib, Seaborn and Streamlit.")
