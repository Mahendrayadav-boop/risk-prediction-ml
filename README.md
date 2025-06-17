# risk-prediction-ml
This project builds a machine learning-based financial risk scoring engine that predicts the likelihood of a customer defaulting on a loan. It leverages historical financial and behavioral data to power informed lending decisions.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap
import joblib

# Load data
df = pd.read_csv("data/credit_data.csv")
X = df.drop("default", axis=1)
y = df["default"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Explain
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test)

# Save
joblib.dump(model, "models/rf_credit_model.pkl")
