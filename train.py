import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import xgboost as xgb
import json
import os

def preprocess_data():
    df = sns.load_dataset("penguins").dropna()

    # Encode target
    le = LabelEncoder()
    df["species"] = le.fit_transform(df["species"])

    # One-hot encode features
    df = pd.get_dummies(df, columns=["sex", "island"])

    X = df.drop(columns=["species"])
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, le, X.columns.tolist()

def train_model(X_train, y_train):
    model = xgb.XGBClassifier(max_depth=3, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    preds = model.predict(X)
    return f1_score(y, preds, average='weighted')

def save_model(model, label_encoder, feature_columns):
    os.makedirs("app/data", exist_ok=True)
    model.save_model("app/data/model.json")

    with open("app/data/meta.json", "w") as f:
        json.dump({
            "classes": list(label_encoder.classes_),
            "features": feature_columns
        }, f)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le, feature_cols = preprocess_data()
    model = train_model(X_train, y_train)
    print("Train F1:", evaluate_model(model, X_train, y_train))
    print("Test F1:", evaluate_model(model, X_test, y_test))
    save_model(model, le, feature_cols)
