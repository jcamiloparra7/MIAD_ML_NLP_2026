from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from inference import MODEL_PATH, build_feature_frame

DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "phishing.csv"


def train_model():
    data = pd.read_csv(DATASET_PATH)
    X = build_feature_frame(data["url"])
    y = data["phishing"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH, compress=3)
    return MODEL_PATH


if __name__ == "__main__":
    output_path = train_model()
    print(f"Saved model to {output_path}")
