import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

KSERVE_URL = "http://168.144.71.162:32085/v1/models/sklearn-employee-attrition:predict"
FEAST_SERVER_URL = "http://168.144.71.162:30800"

# Exactly the 19 features the model was trained on.
# Source: _08_training.py drops employee_id, event_timestamp, attrition before fit.
# Source: _06_preprocessing.py ColumnTransformer named columns (no employee_id).
MODEL_FEATURES = [
    "age_group", "annual_income", "company_reputation", "company_size",
    "company_tenure", "education_level", "early_company_tenure_risk",
    "job_level", "long_tenure_low_role_risk", "number_of_dependents",
    "number_of_promotions", "opportunities", "overtime", "overall_satisfaction",
    "performance_rating", "remote_work", "role_stagnation_ratio",
    "tenure_gap", "years_at_company",
]

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/features/<employeeId>", methods=["GET"])
def get_features(employeeId):
    try:
        print(f"\n🔹 Fetching features for employee: {employeeId}")

        resp = requests.post(
            f"{FEAST_SERVER_URL}/get-online-features",
            json={
                "feature_service": "employee_attrition_features",
                "entities": {"employee_id": [int(employeeId)]},
            },
            timeout=10,
        )

        if resp.status_code != 200:
            return jsonify({"error": resp.text}), resp.status_code

        data = resp.json()
        feature_names = data["metadata"]["feature_names"]
        values = [row["values"][0] for row in data["results"]]

        # Feast HTTP API returns names as "featureview__feature_name".
        # Strip the prefix so names match what the model was trained on.
        feast_features = {
            name.split("__", 1)[-1]: val
            for name, val in zip(feature_names, values)
        }

        feast_features.pop("employee_id", None)
        feast_features.pop("attrition", None)

        print(f"✅ Feast features ({len(feast_features)}): {feast_features}")

        return jsonify({"employee_id": employeeId, "features": feast_features})

    except Exception as e:
        print(f"❌ Feature Fetch Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        print(f"\n🔹 Incoming keys: {list(data.keys())}")

        # Build exactly the 19 model features in the order they were named
        # in the ColumnTransformer. employee_id is excluded — the model
        # was trained without it (dropped in _08_training.py before fit).
        row = {}
        missing = []
        for col in MODEL_FEATURES:
            if col in data:
                row[col] = data[col]
            else:
                row[col] = 0
                missing.append(col)

        if missing:
            print(f"⚠️  Missing features (defaulted to 0): {missing}")

        print(f"📦 Sending to KServe: {row}")

        # KServe sklearn server calls pd.DataFrame(instances[i]) on each element.
        # Values must be wrapped in lists so pd.DataFrame builds a named
        # 1-row DataFrame — the model's ColumnTransformer requires named columns.
        kserve_row = {k: [v] for k, v in row.items()}

        response = requests.post(
            KSERVE_URL,
            json={"instances": [kserve_row]},
            timeout=10,
        )

        if response.status_code != 200:
            return jsonify({"error": response.text}), 500

        result = response.json()
        print(f"✅ Prediction: {result}")

        return jsonify({"prediction": result["predictions"][0]})

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return jsonify({"error": str(e)}), 400


# Diagnostic endpoint — use to verify Feast is returning the right values.
# Visit /debug/<employeeId> in a browser to inspect the full pipeline.
@app.route("/debug/<employeeId>", methods=["GET"])
def debug(employeeId):
    try:
        resp = requests.post(
            f"{FEAST_SERVER_URL}/get-online-features",
            json={
                "feature_service": "employee_attrition_features",
                "entities": {"employee_id": [int(employeeId)]},
            },
            timeout=10,
        )
        raw = resp.json()
        feature_names = raw["metadata"]["feature_names"]
        values = [row["values"][0] for row in raw["results"]]

        stripped = {
            name.split("__", 1)[-1]: val
            for name, val in zip(feature_names, values)
        }
        stripped.pop("employee_id", None)
        stripped.pop("attrition", None)

        model_row = {col: stripped.get(col, "MISSING") for col in MODEL_FEATURES}
        missing = [col for col in MODEL_FEATURES if col not in stripped]

        return jsonify({
            "raw_feature_names_from_feast": feature_names,
            "stripped_features": stripped,
            "model_row_to_be_sent": model_row,
            "missing_from_feast": missing,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
