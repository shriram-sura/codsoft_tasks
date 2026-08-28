from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("fraud_detection_pipeline.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "category": request.form["category"],
            "amt": float(request.form["amt"]),
            "gender": request.form["gender"],
            "state": request.form["state"],
            "lat": float(request.form["lat"]),
            "long": float(request.form["long"]),
            "city_pop": float(request.form["city_pop"]),
            "merch_lat": float(request.form["merch_lat"]),
            "merch_long": float(request.form["merch_long"]),
            "trans_hour": int(request.form["trans_hour"]),
            "trans_day_of_week": int(request.form["trans_day_of_week"]),
            "trans_month": int(request.form["trans_month"]),
            "age": int(request.form["age"]),
            "log_amt": float(request.form["log_amt"]),
            "location_distance": float(request.form["location_distance"])
        }

        input_data = pd.DataFrame([data])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "⚠️ Potential Fraudulent Transaction"
        else:
            result = "✅ Legitimate Transaction"

        return render_template(
            "index.html",
            result=result
        )

    except Exception as error:
        return render_template(
            "index.html",
            result=f"Error: {error}"
        )


if __name__ == "__main__":
    app.run(debug=True)