from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_and_save_model(model_path: str):
    diabetes = load_diabetes()
    print("dsfdsf")
    X, y = diabetes.data, diabetes.target
    print(X.shape)
    model = LinearRegression()
    model.fit(X, y)
    print(model)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    full_path = os.path.abspath(model_path)
    print(f"Saving model to: {full_path}")

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model("../models/diabetes_model.pkl")
