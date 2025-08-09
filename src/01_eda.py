#EXPLANATORY DATA ANALYSIS

#importing necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#display settings
pd.set_option("display.max_columns", None)
sns.set(style="darkgrid", palette="muted")

#loading dataset
df = pd.read_csv("../data/raw/ks-projects-201801.csv")
print(df.head())
print(df.shape)

#basic information
print(df.info())
print(df.describe(include="all"))

#DATA CLEANING

#removing duplicates
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

#handling missing values
df.isnull().sum()

#strip whitespace from strings
df = df.applymap(lambda x:x.strip() if isinstance(x, str) else x)

#convert date columns
date_cols = ["deadline", "launched"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

#Univariate Analysis

#categorical variables
for col in ["main_category", "country", "state", "category", "currency"]:
    plt.figure(figsize=(10,5))
    df[col].value_counts().head(20).plot(kind="bar")
    plt.title(f"Top {col} values")
    plt.show()

#numerical variables
num_features = ["goal", "pledged", "backers", "usd_pledged_real", "usd_goal_real"]
for col in num_features:
    plt.figure(figsize=(10,5))
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

#Bivariate Analysis

#funding by status
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="state", y="usd_pledged_real")
plt.title("Funding by Company Status")
plt.show()

#funding by country(top 10)
top_coountries = df["country"].value_counts().head(10).index
plt.figure(figsize=(12,6))
sns.barplot(data=df[df["country"].isin(top_coountries)], x="country", y="usd_pledged_real", estimator=np.median)
plt.title("Median funding by Country")
plt.show()

#success rate by main category
plt.figure(figsize=(12,6))
sns.countplot(data=df, x="main_category", hue="state")
plt.xticks(rotation=45)
plt.title("Project State by Main Category")
plt.show()

#goal vs state
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x="state", y="goal")
plt.ylim(0,50000)
plt.title("Goal amount by Project State")
plt.show()

#Time-based Analysis

#extract year and month
df["launch_year"] = df["launched"].dt.year
df["launch_month"] = df["launched"].dt.month

#projects per year
plt.figure(figsize=(10,5))
sns.countplot(data=df, x="launch_year", hue="state")
plt.title("Projects per Year by State")

#Correlations

num_cols = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10,5))
sns.heatmap(num_cols.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")

#Saving Cleaned Data
df.to_csv("../data/processed/01_eda_cleaned.csv", index=False)
