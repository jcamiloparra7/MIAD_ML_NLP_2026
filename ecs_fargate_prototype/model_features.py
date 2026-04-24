import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

columnas_numericas = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

TEXT_COLUMNS = [
    "track_name",
    "album_name",
    "artists",
    "track_genre",
]

REQUIRED_COLUMNS = TEXT_COLUMNS + columnas_numericas


def coerce_explicit_to_bool(value):
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        raise ValueError("'explicit' must be a boolean.")

    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        normalized_value = value.strip().lower()
        if normalized_value in {"true", "1"}:
            return True
        if normalized_value in {"false", "0"}:
            return False

    raise ValueError("'explicit' must be a boolean.")


def coerce_explicit_to_int(value):
    return int(coerce_explicit_to_bool(value))

TRACK_FLAG_PATTERNS = {
    "track_name_has_dash": r"\s[-–]\s",
    "track_name_has_bracketed_context": r"[\(\)\[\]\"“”]",
    "track_name_has_numbers": r"\b\d+(?:st|nd|rd|th)?\b",
    "track_name_has_feat": r"\b(?:feat\.?|ft\.?|featuring)\b",
    "track_name_has_live": r"\b(?:live|en vivo|ao vivo)\b",
    "track_name_has_from": r"\bfrom\b",
    "track_name_has_mix_or_remix": r"\b(?:mix|remix|rmx)\b",
    "track_name_has_version": r"\bversion\b",
    "track_name_has_remastered": r"\bremaster(?:ed)?\b",
    "track_name_has_original": r"\boriginal\b",
    "track_name_has_christmas": r"\b(?:christmas|navidad|xmas)\b",
    "track_name_has_acoustic": r"\b(?:acoustic|acustic|acustico|acústico|acustica|acústica)\b",
}

ALBUM_FLAG_PATTERNS = {
    "album_name_has_bracketed_context": r"[\(\)\[\]\"“”]",
    "album_name_has_numbers": r"\b\d+(?:st|nd|rd|th)?\b",
    "album_name_has_live": r"\b(?:live|en vivo|ao vivo)\b",
    "album_name_has_volume": r"\b(?:vol\.?|volume|volumen)\s*\d*\b",
    "album_name_has_christmas": r"\b(?:christmas|navidad|xmas)\b",
    "album_name_has_soundtrack": r"\b(?:soundtrack|ost|bso)\b|original motion picture|motion picture soundtrack",
    "album_name_has_edition": r"\b(?:edition|edicion|edición|édition)\b",
    "album_name_has_deluxe": r"\bdeluxe\b",
    "album_name_has_remaster": r"\bremaster(?:ed)?\b",
    "album_name_has_version": r"\bversion\b",
    "album_name_has_acoustic": r"\b(?:acoustic|acustic|acustico|acústico|acustica|acústica)\b",
    "album_name_has_ep": r"(?<![a-z])ep(?![a-z])",
    "album_name_has_anniversary": r"\b(?:anniversary|aniversario)\b",
}


class SpotifyFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        numeric_columns,
        artist_popularity_smoothing=2,
        genre_popularity_smoothing=2,
        album_popularity_smoothing=10,
        primary_artist_genre_smoothing=2,
        album_genre_smoothing=20,
    ):
        self.numeric_columns = numeric_columns
        self.artist_popularity_smoothing = artist_popularity_smoothing
        self.genre_popularity_smoothing = genre_popularity_smoothing
        self.album_popularity_smoothing = album_popularity_smoothing
        self.primary_artist_genre_smoothing = primary_artist_genre_smoothing
        self.album_genre_smoothing = album_genre_smoothing

    @staticmethod
    def _parse_artists(artist_value):
        if pd.isna(artist_value):
            return []
        normalized_text = str(artist_value).replace(";", ",")
        artists = [artist.strip() for artist in normalized_text.split(",") if artist.strip()]
        return artists

    @staticmethod
    def _primary_artist(artist_list):
        return artist_list[0] if artist_list else "missing"

    @staticmethod
    def _safe_text(value):
        if pd.isna(value):
            return ""
        return str(value).strip().lower()

    @staticmethod
    def _album_key(value):
        return "missing" if pd.isna(value) else str(value).strip() or "missing"

    @staticmethod
    def _artist_genre_key(primary_artist, genre):
        normalized_genre = "missing" if pd.isna(genre) else genre
        return (primary_artist, normalized_genre)

    @staticmethod
    def _album_genre_key(album, genre):
        normalized_genre = "missing" if pd.isna(genre) else genre
        return (album, normalized_genre)

    def fit(self, X, y=None):
        artist_lists = X["artists"].apply(self._parse_artists)
        primary_artist_values = artist_lists.apply(self._primary_artist)
        genre_values = X["track_genre"].fillna("missing")
        album_values = X["album_name"].apply(self._album_key)

        popularity_sum_by_artist = {}
        support_count_by_artist = {}
        popularity_sum_by_genre = {}
        popularity_count_by_genre = {}
        popularity_sum_by_album = {}
        popularity_count_by_album = {}
        popularity_sum_by_primary_artist_genre = {}
        popularity_count_by_primary_artist_genre = {}
        popularity_sum_by_album_genre = {}
        popularity_count_by_album_genre = {}
        self.global_popularity_ = float(pd.Series(y).mean()) if y is not None else 0.0

        if y is not None:
            y_series = pd.Series(y, index=X.index)
        else:
            y_series = None

        for index, artist_list in artist_lists.items():
            current_target = float(y_series.loc[index]) if y_series is not None else None
            for artist in artist_list:
                support_count_by_artist[artist] = support_count_by_artist.get(artist, 0) + 1
                if current_target is not None:
                    popularity_sum_by_artist[artist] = popularity_sum_by_artist.get(artist, 0.0) + current_target

        if y_series is not None:
            for index in X.index:
                current_target = float(y_series.loc[index])
                genre = genre_values.loc[index]
                album = album_values.loc[index]
                primary_artist = primary_artist_values.loc[index]

                popularity_sum_by_genre[genre] = popularity_sum_by_genre.get(genre, 0.0) + current_target
                popularity_count_by_genre[genre] = popularity_count_by_genre.get(genre, 0) + 1

                popularity_sum_by_album[album] = popularity_sum_by_album.get(album, 0.0) + current_target
                popularity_count_by_album[album] = popularity_count_by_album.get(album, 0) + 1

                artist_genre_key = self._artist_genre_key(primary_artist, genre)
                popularity_sum_by_primary_artist_genre[artist_genre_key] = (
                    popularity_sum_by_primary_artist_genre.get(artist_genre_key, 0.0) + current_target
                )
                popularity_count_by_primary_artist_genre[artist_genre_key] = (
                    popularity_count_by_primary_artist_genre.get(artist_genre_key, 0) + 1
                )

                album_genre_key = self._album_genre_key(album, genre)
                popularity_sum_by_album_genre[album_genre_key] = (
                    popularity_sum_by_album_genre.get(album_genre_key, 0.0) + current_target
                )
                popularity_count_by_album_genre[album_genre_key] = (
                    popularity_count_by_album_genre.get(album_genre_key, 0) + 1
                )

        self.artist_support_dict_ = {
            artist: float(artist_count)
            for artist, artist_count in support_count_by_artist.items()
        }
        self.artist_popularity_dict_ = {}
        for artist, artist_count in support_count_by_artist.items():
            artist_mean = popularity_sum_by_artist[artist] / artist_count
            smoothed_score = (
                artist_count * artist_mean + self.artist_popularity_smoothing * self.global_popularity_
            ) / (artist_count + self.artist_popularity_smoothing)
            self.artist_popularity_dict_[artist] = float(smoothed_score)

        self.genre_popularity_dict_ = {}
        for genre, genre_count in popularity_count_by_genre.items():
            genre_mean = popularity_sum_by_genre[genre] / genre_count
            smoothed_score = (
                genre_count * genre_mean + self.genre_popularity_smoothing * self.global_popularity_
            ) / (genre_count + self.genre_popularity_smoothing)
            self.genre_popularity_dict_[genre] = float(smoothed_score)

        self.album_popularity_dict_ = {}
        for album, album_count in popularity_count_by_album.items():
            album_mean = popularity_sum_by_album[album] / album_count
            smoothed_score = (
                album_count * album_mean + self.album_popularity_smoothing * self.global_popularity_
            ) / (album_count + self.album_popularity_smoothing)
            self.album_popularity_dict_[album] = float(smoothed_score)

        self.primary_artist_genre_popularity_dict_ = {}
        for interaction_key, interaction_count in popularity_count_by_primary_artist_genre.items():
            primary_artist, genre = interaction_key
            interaction_mean = popularity_sum_by_primary_artist_genre[interaction_key] / interaction_count
            hierarchical_prior = (self._artist_score(primary_artist) + self._genre_score(genre)) / 2
            smoothed_score = (
                interaction_count * interaction_mean + self.primary_artist_genre_smoothing * hierarchical_prior
            ) / (interaction_count + self.primary_artist_genre_smoothing)
            self.primary_artist_genre_popularity_dict_[interaction_key] = float(smoothed_score)

        self.album_genre_popularity_dict_ = {}
        for interaction_key, interaction_count in popularity_count_by_album_genre.items():
            album, genre = interaction_key
            interaction_mean = popularity_sum_by_album_genre[interaction_key] / interaction_count
            hierarchical_prior = (self._album_score(album) + self._genre_score(genre)) / 2
            smoothed_score = (
                interaction_count * interaction_mean + self.album_genre_smoothing * hierarchical_prior
            ) / (interaction_count + self.album_genre_smoothing)
            self.album_genre_popularity_dict_[interaction_key] = float(smoothed_score)

        return self

    def _artist_scores(self, artist_list):
        return [self._artist_score(artist) for artist in artist_list]

    def _artist_supports(self, artist_list):
        return [self._artist_support(artist) for artist in artist_list]

    def _artist_score(self, artist):
        return float(self.artist_popularity_dict_.get(artist, self.global_popularity_))

    def _artist_support(self, artist):
        return float(self.artist_support_dict_.get(artist, 0))

    def _genre_score(self, genre):
        normalized_genre = "missing" if pd.isna(genre) else genre
        return float(self.genre_popularity_dict_.get(normalized_genre, self.global_popularity_))

    def _primary_artist_genre_score(self, primary_artist, genre):
        interaction_key = self._artist_genre_key(primary_artist, genre)
        hierarchical_prior = (self._artist_score(primary_artist) + self._genre_score(genre)) / 2
        return float(self.primary_artist_genre_popularity_dict_.get(interaction_key, hierarchical_prior))

    def _album_genre_score(self, album, genre):
        normalized_album = self._album_key(album)
        interaction_key = self._album_genre_key(normalized_album, genre)
        hierarchical_prior = (self._album_score(normalized_album) + self._genre_score(genre)) / 2
        return float(self.album_genre_popularity_dict_.get(interaction_key, hierarchical_prior))

    def _album_score(self, album):
        normalized_album = self._album_key(album)
        return float(self.album_popularity_dict_.get(normalized_album, self.global_popularity_))

    def _artist_popularity_with_support(self, artist_list):
        if not artist_list:
            return self.global_popularity_, 0.0

        best_artist = None
        best_score = float("-inf")
        for artist in artist_list:
            current_score = self._artist_score(artist)
            if current_score > best_score:
                best_score = current_score
                best_artist = artist

        best_support = self._artist_support(best_artist)
        return float(best_score), float(best_support)

    def _primary_artist_with_support(self, artist_list):
        if not artist_list:
            return self.global_popularity_, 0.0
        primary_artist = artist_list[0]
        return self._artist_score(primary_artist), self._artist_support(primary_artist)

    def _ordered_artist_aggregates(self, artist_list):
        if not artist_list:
            return self.global_popularity_, 0.0

        scores = np.array(self._artist_scores(artist_list), dtype=float)
        supports = np.array(self._artist_supports(artist_list), dtype=float)
        weights = np.array([1.0 / (index + 1) for index in range(len(artist_list))], dtype=float)

        ordered_score = float(np.average(scores, weights=weights))
        ordered_support = float(np.average(supports, weights=weights))
        return ordered_score, ordered_support

    def _artist_gap_primary_vs_best(self, artist_list):
        if not artist_list:
            return 0.0
        primary_score = self._artist_score(artist_list[0])
        best_score = max(self._artist_scores(artist_list))
        return float(primary_score - best_score)

    def transform(self, X):
        transformed = pd.DataFrame(index=X.index)

        numeric_frame = X[self.numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        transformed[self.numeric_columns] = numeric_frame

        artist_lists = X["artists"].apply(self._parse_artists)
        primary_artist_values = artist_lists.apply(self._primary_artist)
        genre_values = X["track_genre"].fillna("missing")
        album_values = X["album_name"].apply(self._album_key)

        artist_score_support = artist_lists.apply(self._artist_popularity_with_support)
        transformed["artist_popularity_score"] = artist_score_support.apply(lambda value: value[0])
        transformed["artist_support_for_max"] = artist_score_support.apply(lambda value: value[1])

        primary_artist_features = artist_lists.apply(self._primary_artist_with_support)
        transformed["artist_popularity_primary"] = primary_artist_features.apply(lambda value: value[0])
        transformed["primary_artist_genre_popularity_score"] = [
            self._primary_artist_genre_score(primary_artist, genre)
            for primary_artist, genre in zip(primary_artist_values, genre_values)
        ]
        transformed["album_genre_popularity_score"] = [
            self._album_genre_score(album, genre)
            for album, genre in zip(album_values, genre_values)
        ]

        ordered_artist_features = artist_lists.apply(self._ordered_artist_aggregates)
        transformed["artist_popularity_ordered_mean"] = ordered_artist_features.apply(lambda value: value[0])
        transformed["artist_support_ordered_mean"] = ordered_artist_features.apply(lambda value: value[1])

        transformed["artist_gap_primary_vs_best"] = artist_lists.apply(self._artist_gap_primary_vs_best)

        track_names = X["track_name"].apply(self._safe_text)
        for feature_name, pattern in TRACK_FLAG_PATTERNS.items():
            transformed[feature_name] = track_names.str.contains(pattern, regex=True).astype(float)

        album_names = X["album_name"].apply(self._safe_text)
        for feature_name, pattern in ALBUM_FLAG_PATTERNS.items():
            transformed[feature_name] = album_names.str.contains(pattern, regex=True).astype(float)

        transformed["genre_popularity_score"] = genre_values.apply(self._genre_score)
        transformed["album_popularity_score"] = album_values.apply(self._album_score)
        transformed["primary_artist_genre_popularity_delta_vs_genre"] = (
            transformed["primary_artist_genre_popularity_score"] - transformed["genre_popularity_score"]
        )
        transformed["album_genre_popularity_delta_vs_album"] = (
            transformed["album_genre_popularity_score"] - transformed["album_popularity_score"]
        )

        return transformed


engineered_numeric_columns = columnas_numericas + [
    "artist_popularity_score",
    "artist_support_for_max",
    "artist_popularity_primary",
    "primary_artist_genre_popularity_score",
    "album_genre_popularity_score",
    "primary_artist_genre_popularity_delta_vs_genre",
    "album_genre_popularity_delta_vs_album",
    "artist_popularity_ordered_mean",
    "artist_support_ordered_mean",
    "artist_gap_primary_vs_best",
    "track_name_has_dash",
    "track_name_has_bracketed_context",
    "track_name_has_numbers",
    "track_name_has_feat",
    "track_name_has_live",
    "track_name_has_from",
    "track_name_has_mix_or_remix",
    "track_name_has_version",
    "track_name_has_remastered",
    "track_name_has_original",
    "track_name_has_christmas",
    "track_name_has_acoustic",
    "album_name_has_bracketed_context",
    "album_name_has_numbers",
    "album_name_has_live",
    "album_name_has_volume",
    "album_name_has_christmas",
    "album_name_has_soundtrack",
    "album_name_has_edition",
    "album_name_has_deluxe",
    "album_name_has_remaster",
    "album_name_has_version",
    "album_name_has_acoustic",
    "album_name_has_ep",
    "album_name_has_anniversary",
    "genre_popularity_score",
    "album_popularity_score",
]
