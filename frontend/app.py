import requests
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

KSERVE_URL = os.environ.get("KSERVE_URL", "http://localhost:7070/v1/models/employee_attrition_prediction:predict")
FEAST_SERVER_URL = os.environ.get("FEAST_SERVER_URL", "http://feast-server.feast.svc.cluster.local:6566")

# Must match the column order the model was trained on (entity + label excluded)
FEATURE_COLUMNS = [
    "age_group", "annual_income", "company_reputation", "company_size",
    "company_tenure", "education_level", "early_company_tenure_risk",
    "job_level", "long_tenure_low_role_risk", "number_of_dependents",
    "number_of_promotions", "opportunities", "overtime",
    "overall_satisfaction", "performance_rating", "remote_work",
    "role_stagnation_ratio", "tenure_gap", "years_at_company",
]

print("Model features:", FEATURE_COLUMNS)

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')



@app.route("/features/<employeeId>", methods=["GET"])
def get_features(employeeId):
    print(employeeId)

    resp = requests.post(
        f"{FEAST_SERVER_URL}/get-online-features",
        json={
            "feature_service": "employee_attrition_features",
            "entities": {"employee_id": [int(employeeId)]},
        },
        timeout=10,
    )

    if resp.status_code != 200:
        return jsonify({"error": f"Feast Server error: {resp.text}"}), resp.status_code

    data = resp.json()
    feature_names = data["metadata"]["feature_names"]
    values = data["results"][0]["values"]
    feast_features = dict(zip(feature_names, values))

    print('feast-response: ', feast_features)

    if all(v is None for k, v in feast_features.items() if k != "employee_id"):
        return jsonify({"error": f"Employee {employeeId} not found"}), 404

    feast_features.pop("employee_id", None)
    feast_features.pop("attrition", None)

    return jsonify({
        "employee_id": employeeId,
        "features": feast_features
    })

    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data.pop("employee_id", None)
    print('incoming-data: ', data)

    # Engineered features (role_stagnation_ratio, tenure_gap, etc.)
    # are already stored in Feast and returned by get_features — no recomputation needed

    try:
        missing = set(FEATURE_COLUMNS) - set(data.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")

        df_input = pd.DataFrame([data]).reindex(columns=FEATURE_COLUMNS)
        print('df-input: ', df_input.to_dict(orient="records"))

        response = requests.post(
            KSERVE_URL,
            json={"instances": df_input.values.tolist()}
        )
        print('results: ', response.json())

        prediction_result = response.json()
        # { predictions: [1] }

        return jsonify({"prediction": prediction_result['predictions'][0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)