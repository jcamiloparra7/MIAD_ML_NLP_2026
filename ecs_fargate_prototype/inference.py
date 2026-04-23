from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).with_name("phishing_clf.pkl")
KEYWORDS = ("https", "login", ".php", ".html", "@", "sign")


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def build_feature_frame(urls):
    url_frame = pd.DataFrame({"url": pd.Series(urls, dtype="string")})

    for keyword in KEYWORDS:
        url_frame[f"keyword_{keyword}"] = url_frame.url.str.contains(keyword).astype(int)

    url_frame["lenght"] = url_frame.url.str.len() - 2
    domain = url_frame.url.str.split("/", expand=True).iloc[:, 2]
    url_frame["lenght_domain"] = domain.str.len()
    url_frame["isIP"] = url_frame.url.str.replace(".", "", regex=False).str.isnumeric().astype(int)
    url_frame["count_com"] = url_frame.url.str.count("com")

    return url_frame.drop("url", axis=1)


def predict_proba(url):
    clf = load_model()
    features = build_feature_frame([url])
    return float(clf.predict_proba(features)[0, 1])
