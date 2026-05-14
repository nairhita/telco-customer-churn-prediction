# telco-customer-churn-prediction
End-to-end churn prediction pipeline using XGBoost and SHAP for model interpretability.


📊 Telco Customer Churn PredictionPredicting high-value customer attrition using XGBoost and explainable AI techniques.

📌 Project OverviewIn the telecommunications industry, customer retention is often more cost-effective than acquisition. This project develops a machine learning pipeline to identify customers at high risk of churning. By predicting churn, businesses can proactively engage customers with retention offers, potentially saving millions in lost revenue.

🎯 Business ObjectivePrimary Goal: Identify at-risk customers with high Recall (minimizing False Negatives).Key Question: What are the primary drivers of churn (e.g., contract type, monthly charges, or support issues)?🛠️ Tech StackLanguage: Python 3.xData Manipulation: pandas, numpyVisualization: seaborn, matplotlibModeling: XGBoost, scikit-learnEvaluation: Precision-Recall curves, ROC-AUC, Confusion Matrix

🧬 Data Pipeline & EngineeringThe dataset used is the Kaggle Telco Churn Dataset.
1. Data Cleaning & ConstraintsType Casting: Converted TotalCharges from object to float, handling missing values via median imputation.Feature Selection: Dropped customerID to prevent data leakage and noise.Encoding: Applied Label Encoding for binary features and One-Hot Encoding for multi-categorical features (Internet Service, Contract, etc.).
2. Handling Class ImbalanceThe target variable (Churn) is imbalanced (~26% Churn vs ~74% No Churn).Strategy: Used scale_pos_weight within the XGBoost classifier to penalize misclassifications of the minority class.

📈 Model PerformanceThe final model was evaluated on a 20% hold-out test set.MetricScoreROC-AUC0.84Accuracy0.79F1-Score (Churn)0.61Note on Metrics: While accuracy is 79%, we focused on Recall for the Churn class to ensure the business doesn't miss potential "leavers."
🔍 Key Insights & Feature ImportanceBased on the model's feature importance, the top drivers of churn are:Contract Type: Month-to-month contracts are significantly more likely to churn.Tenure: Customers in their first 6 months have the highest attrition risk.Total Charges: High monthly spending without long-term commitment is a major red flag.

🚀 How to RunClone the repo:Bashgit clone https://github.com/yourusername/telco-churn-prediction.git
Install dependencies:Bash   pip install -r requirements.txt
Run the analysis:Bashpython main.py
🔮 Future WorkHyperparameter Tuning: Implement Optuna or GridSearchCV for more granular model optimization.Explainability: Integrate SHAP (SHapley Additive exPlanations) values to provide per-customer churn reasons.Deployment: Wrap the model in a FastAPI or Flask wrapper for real-time predictions.
