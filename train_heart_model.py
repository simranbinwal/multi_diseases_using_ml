import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("datasets/heart.csv")

X = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang']]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, pipeline.predict(X_test))
print("Heart Disease Model Accuracy:", accuracy)

# Save model
joblib.dump(pipeline, "models/heart_model.pkl")
