#PREPROCESSING

#importing and loading necessary files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

df = pd.read_csv("../data/processed/01_eda_cleaned.csv")
df.head()

#Handling missing values

df.dropna(subset=["name", "category", "main_category", "state"], inplace=True)

# Convert dates to datetime
df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
df['launched'] = pd.to_datetime(df['launched'], errors='coerce')

#Feature Engineering

#create campaign duration
df["campaign_days"] = (df["deadline"] - df["launched"]).dt.days

#target variable: success/fail (binary)
df = df[df["state"].isin(["successful", "failed"])]
df["target"] = df["state"].map({"successful":1, "failed":0})

#Encoding Categorical Variables

#label encoding
label_enc_cols = ["category", "main_category", "currency", "country"]
label_encoders = {}
for col in label_enc_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

#Scaling Numerical Features
num_features = ["goal", "pledged", "backers", "usd_pledged_real", "usd_goal_real", "campaign_days"]
scaler = StandardScaler()
df[num_features] = scaler.fit_transform(df[num_features])

#Train-Test Split
X = df.drop(columns=["ID", "name", "state", "target", "deadline", "launched"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

#Saving Processed Data
X_train.to_csv("../data/processed/X_train.csv", index=False)
y_train.to_csv("../data/processed/y_train.csv", index=False)
X_test.to_csv("../data/processed/X_test.csv", index=False)
y_test.to_csv("../data/processed/y_test.csv", index=False)