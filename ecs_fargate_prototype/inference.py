from functools import lru_cache

import joblib
import pandas as pd

from model_features import REQUIRED_COLUMNS, coerce_explicit_to_int
from train_model import MODEL_PATH


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def build_feature_frame(records):
    feature_frame = pd.DataFrame.from_records(records)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in feature_frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required fields: {missing_columns}")

    feature_frame = feature_frame[REQUIRED_COLUMNS].copy()
    feature_frame["explicit"] = feature_frame["explicit"].apply(coerce_explicit_to_int)
    return feature_frame


def predict_popularity(records):
    model = load_model()
    features = build_feature_frame(records)
    predictions = model.predict(features)
    return [float(value) for value in predictions]
