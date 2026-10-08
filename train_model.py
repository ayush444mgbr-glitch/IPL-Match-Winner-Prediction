import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ipl.csv"
MODEL_PATH = BASE_DIR / "model" / "ipl_winner_model.joblib"
META_PATH = BASE_DIR / "model" / "metadata.json"

FEATURES = [
    "team1", "team2", "toss_winner", "toss_decision",
    "venue", "city", "season", "match_type"
]
TARGET = "team1_won"

# Different names used by the same franchise across IPL seasons are grouped together.
TEAM_ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",
    "Delhi Daredevils": "Delhi Capitals",
    "Delhi Capitals": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Punjab Kings": "Punjab Kings",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}


def canonical_team(value):
    return TEAM_ALIASES.get(value, value)


def build_dataset():
    df = pd.read_csv(DATA_PATH)

    data = df[FEATURES + ["winner"]].dropna(subset=["winner"]).copy()

    for col in ["team1", "team2", "toss_winner"]:
        data[col] = data[col].map(canonical_team)

    # A missing city is kept as a separate category instead of dropping the match.
    data["city"] = data["city"].fillna("Unknown")
    data["venue"] = data["venue"].fillna("Unknown")
    data["toss_decision"] = data["toss_decision"].fillna("Unknown")
    data["match_type"] = data["match_type"].fillna("Unknown")

    data["winner"] = data["winner"].map(canonical_team)

    # Binary target: for a selected Team 1 vs Team 2 match, the model predicts
    # whether Team 1 wins (1) or Team 2 wins (0). This prevents unrelated teams
    # from appearing in the final head-to-head result.
    data = data[(data["winner"] == data["team1"]) | (data["winner"] == data["team2"])].copy()
    data["team1_won"] = (data["winner"] == data["team1"]).astype(int)

    return df, data


def train():
    raw_df, data = build_dataset()

    X = data[FEATURES].copy()
    y = data[TARGET].copy()

    categorical_features = [
        "team1", "team2", "toss_winner", "toss_decision",
        "venue", "city", "match_type"
    ]
    numeric_features = ["season"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    base_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    param_grid = {
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train_encoded, y_train)

    model = grid.best_estimator_
    predictions = model.predict(X_test_encoded)
    probabilities = model.predict_proba(X_test_encoded)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)
    cm = confusion_matrix(y_test, predictions).tolist()

    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "features": FEATURES,
        "team_aliases": TEAM_ALIASES,
        "teams": sorted(set(data["team1"]).union(set(data["team2"]))),
        "venues": sorted(data["venue"].dropna().unique().tolist()),
        "cities": sorted(data["city"].dropna().unique().tolist()),
        "seasons": sorted(data["season"].dropna().unique().tolist()),
        "match_types": sorted(data["match_type"].dropna().unique().tolist()),
        "target_description": "1 = Team 1 wins, 0 = Team 2 wins",
    }
    joblib.dump(bundle, MODEL_PATH)

    metadata = {
        "original_rows": int(len(raw_df)),
        "rows_used_for_model": int(len(data)),
        "features": FEATURES,
        "target": TARGET,
        "target_description": "1 = Team 1 wins, 0 = Team 2 wins",
        "team_count": int(len(bundle["teams"])),
        "teams": bundle["teams"],
        "test_accuracy": round(float(accuracy), 4),
        "test_roc_auc": round(float(auc), 4),
        "best_parameters": grid.best_params_,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Model training completed.")
    print(f"Rows used: {len(data)}")
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Test ROC-AUC: {auc:.3f}")
    print(f"Best parameters: {grid.best_params_}")
    print(f"Saved model: {MODEL_PATH}")

    return metadata


if __name__ == "__main__":
    train()
