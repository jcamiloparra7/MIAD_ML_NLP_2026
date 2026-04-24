from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from model_features import (
    SpotifyFeatureBuilder,
    columnas_numericas,
    engineered_numeric_columns,
)


TRAIN_URL = 'https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2026/main/datasets/dataTrain_Spotify.csv'
MODEL_PATH = Path(__file__).with_name("spotify_popularity_rf.pkl")

BEST_ALPHA_CONFIG = {
    'artist_alpha': 2,
    'genre_alpha': 2,
    'album_alpha': 10,
    'primary_artist_genre_alpha': 2,
    'album_genre_alpha': 20,
}
FINAL_RF_ESTIMATORS = 250
def build_random_forest_pipeline(alpha_config=None, n_estimators=FINAL_RF_ESTIMATORS):
    if alpha_config is None:
        alpha_config = BEST_ALPHA_CONFIG

    preprocessor = ColumnTransformer(
        transformers=[('numeric', 'passthrough', engineered_numeric_columns)],
        remainder='drop'
    )

    return Pipeline(
        steps=[
            ('feature_builder', SpotifyFeatureBuilder(
                columnas_numericas,
                artist_popularity_smoothing=alpha_config['artist_alpha'],
                genre_popularity_smoothing=alpha_config['genre_alpha'],
                album_popularity_smoothing=alpha_config['album_alpha'],
                primary_artist_genre_smoothing=alpha_config['primary_artist_genre_alpha'],
                album_genre_smoothing=alpha_config['album_genre_alpha'],
            )),
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=n_estimators,
                n_jobs=-1,
                random_state=42,
                max_features='log2',
            )),
        ]
    )


def train_model():
    data = pd.read_csv(TRAIN_URL)
    X = data.drop(columns=['popularity'])
    y = data['popularity']
    model = build_random_forest_pipeline()    
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH, compress=3)
    return MODEL_PATH


if __name__ == "__main__":
    output_path = train_model()
    print(f"Saved model to {output_path}")
