import os
import numpy as np
import librosa

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib


def initialize_dataset():
    file_paths, labels = [], []
    NOT_TRAIN_DIR = (
        "/Users/joeschlessinger/Programming/trainTrackER/recordings/not_train"
    )
    TRAIN_DIR = "/Users/joeschlessinger/Programming/trainTrackER/recordings/train"

    for file in [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR)]:
        if file.endswith(".wav"):
            file_paths.append(file)
            labels.append(1)

    for file in [os.path.join(NOT_TRAIN_DIR, f) for f in os.listdir(NOT_TRAIN_DIR)]:
        if file.endswith(".wav"):
            file_paths.append(file)
            labels.append(0)

    return file_paths, labels


def peak_length(y, sr, threshold_factor=1.5):
    # Calculate the root mean square (RMS) amplitude
    rms = librosa.feature.rms(y=y)[0]

    # Apply thresholding
    threshold = np.mean(rms) + 1.5 * np.std(rms)

    # threshold = np.mean(rms) * threshold_factor
    segments = np.where(rms > threshold)[0]

    # Select target segment
    if len(segments) > 0:
        start_frame = segments[0]
        end_frame = segments[-1]
        start_time = librosa.frames_to_time(start_frame, sr=sr)
        end_time = librosa.frames_to_time(end_frame, sr=sr)
        return end_time - start_time
    else:
        return 0


def extract_features(file_path, n_bins=10, min_freq=0, max_freq=1000):
    # takes file and outputs X for the model
    y, sr = librosa.load(file_path, sr=None)

    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    target_freqs = freqs[(freqs >= min_freq) & (freqs <= max_freq)]
    target_stft = stft[(freqs >= min_freq) & (freqs <= max_freq)]
    bin_edges = np.linspace(target_freqs[0], target_freqs[-1], n_bins + 1)
    mean_features = []
    std_features = []
    for i in range(n_bins):
        bin_mask = (target_freqs >= bin_edges[i]) & (target_freqs < bin_edges[i + 1])
        bin_stft = target_stft[bin_mask]
        mean_features.append(np.mean(bin_stft))
        std_features.append(np.std(bin_stft))
    # features = np.hstack([mean_features, std_features, [peak_length(y, sr)]])
    features = np.hstack([mean_features, std_features])
    return features


class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=10, min_freq=0, max_freq=1000):
        self.n_bins = n_bins
        self.min_freq = min_freq
        self.max_freq = max_freq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(
            [
                extract_features(fp, self.n_bins, self.min_freq, self.max_freq)
                for fp in X
            ]
        )


def train_model_explicit(n_bins, min_freq, max_freq, save=False, file_name=None):
    file_paths, labels = initialize_dataset()

    X = np.array(
        [
            extract_features(fp, n_bins=n_bins, min_freq=min_freq, max_freq=max_freq)
            for fp in file_paths
        ]
    )
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    if save:
        joblib.dump(clf, f"./{file_name}.pkl")


def train_model_grid_search():
    file_paths, labels = initialize_dataset()
    X = np.array([extract_features(fp) for fp in file_paths])
    y = np.array(labels)
    print(f"Dataset initialized with {len(labels)} data points")
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42
    )

    pipe = Pipeline(
        [
            ("feature_extractor", CustomFeatureExtractor()),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    param_grid = {
        "feature_extractor__n_bins": list(range(5, 41, 5)),
        "feature_extractor__min_freq": list(range(0, 501, 100)),
        "feature_extractor__max_freq": list(range(500, 1001, 100)),
    }

    grid_search = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(n_splits=5))

    grid_search.fit(X_train_paths, y_train)

    print("Best parameters found: ", grid_search.best_params_)

    y_pred = grid_search.best_estimator_.predict(X_test_paths)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return grid_search.best_estimator_


def get_best_model():
    best_model = joblib.load("./best_model.pkl")


def classify_audio(file_path, model):
    return model.predict([file_path])[0]


if __name__ == "__main__":
    best_model = train_model_grid_search()

    joblib.dump(best_model, "./best_model.pkl")
