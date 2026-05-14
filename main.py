import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

# --- 1. Data Loading ---
# Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# --- 2. Data Cleaning ---
# TotalCharges is read as an object but is numeric. 
# We use errors='coerce' to turn blanks into NaNs.
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True) # Drop the ~11 rows with missing TotalCharges

# Drop customerID - it has no predictive power
df.drop('customerID', axis=1, inplace=True)

# --- 3. Feature Engineering & Encoding ---
# Binary encoding for 'Yes'/'No' columns
binary_cols = [col for col in df.columns if df[col].dtype == 'O' and df[col].nunique() == 2]
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-Hot Encoding for multi-category columns (e.g., InternetService)
df = pd.get_dummies(df)

# --- 4. Train/Test Split ---
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 5. Scaling (Important for some models, good practice) ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- 6. Model Training (XGBoost) ---
# We use scale_pos_weight to handle class imbalance (if any)
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# --- 7. Evaluation ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("### Classification Report ###")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")


importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Drivers of Customer Churn")
plt.show()

import shap

# --- Model Explainability with SHAP ---
# Initialize the SHAP Explainer (TreeExplainer is optimized for XGBoost)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# A. Global Interpretability: Summary Plot
# This shows the magnitude and direction of each feature's impact
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
plt.title("SHAP Global Feature Importance")
plt.show()

# B. Local Interpretability: Single Customer Explanation
# Let's look at the first customer in the test set
# (Note: Use .iloc[0] if X_test is a DataFrame, or index 0 if it's an array)
shap.initjs() # Required for notebook visualization
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test[0,:], feature_names=X.columns)
