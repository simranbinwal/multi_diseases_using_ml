import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv("datasets/stress.csv")

# Features
X = data[
    [
        "Age",
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Heart Rate",
        "Daily Steps"
    ]
]

# Target (numeric stress score)
y = data["Stress Level"]  

# Pipeline: scaling + model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X, y)

# Save model
joblib.dump(pipeline, "models/stress_model.pkl")

print("Stress model trained correctly (numeric output)")
