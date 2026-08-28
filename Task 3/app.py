from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("customer_churn_pipeline.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "CreditScore": float(request.form["CreditScore"]),
            "Geography": request.form["Geography"],
            "Gender": request.form["Gender"],
            "Age": float(request.form["Age"]),
            "Tenure": float(request.form["Tenure"]),
            "Balance": float(request.form["Balance"]),
            "NumOfProducts": float(request.form["NumOfProducts"]),
            "HasCrCard": int(request.form["HasCrCard"]),
            "IsActiveMember": int(request.form["IsActiveMember"]),
            "EstimatedSalary": float(request.form["EstimatedSalary"])
        }

        input_data = pd.DataFrame([data])

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "⚠️ Likely to Churn"
        else:
            result = "✅ Likely to Stay"

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
    app.run(debug=True, port=5001)