from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "ipl_winner_model.joblib"

app = Flask(__name__)
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
preprocessor = bundle["preprocessor"]
features = bundle["features"]
teams = bundle["teams"]

# These lists are stored in the model bundle so the app does not depend on
# reading the training CSV every time it starts.
venues = bundle.get("venues", [])
cities = bundle.get("cities", [])
seasons = bundle.get("seasons", [])
match_types = bundle.get("match_types", [])


@app.route("/")
def home():
    return render_template(
        "index.html",
        teams=teams,
        venues=venues,
        cities=cities,
        seasons=seasons,
        match_types=match_types,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        required = [
            "team1", "team2", "toss_winner", "toss_decision",
            "venue", "city", "season", "match_type"
        ]
        missing = [field for field in required if not str(data.get(field, "")).strip()]
        if missing:
            return jsonify({"error": "Please fill in all the match details."}), 400

        team1 = data["team1"]
        team2 = data["team2"]
        toss_winner = data["toss_winner"]

        if team1 == team2:
            return jsonify({"error": "Team 1 and Team 2 must be different."}), 400

        if toss_winner not in {team1, team2}:
            return jsonify({"error": "Toss winner must be either Team 1 or Team 2."}), 400

        try:
            season = int(data["season"])
        except (TypeError, ValueError):
            return jsonify({"error": "Please select a valid season."}), 400

        row = pd.DataFrame([{
            "team1": team1,
            "team2": team2,
            "toss_winner": toss_winner,
            "toss_decision": data["toss_decision"],
            "venue": data["venue"].strip(),
            "city": data["city"].strip(),
            "season": season,
            "match_type": data["match_type"],
        }], columns=features)

        encoded = preprocessor.transform(row)
        probability_team1 = float(model.predict_proba(encoded)[0][1])
        probability_team2 = 1.0 - probability_team1

        if probability_team1 >= probability_team2:
            winner = team1
        else:
            winner = team2

        return jsonify({
            "winner": winner,
            "team1": team1,
            "team2": team2,
            "team1_probability": round(probability_team1 * 100, 1),
            "team2_probability": round(probability_team2 * 100, 1),
        })

    except Exception as exc:
        return jsonify({"error": f"Prediction could not be completed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
