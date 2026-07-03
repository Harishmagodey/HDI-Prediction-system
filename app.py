from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and imputer
model = joblib.load("model/hdi_model.pkl")
imputer = joblib.load("model/imputer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        life_expectancy = float(request.form["life_expectancy"])
        schooling = float(request.form["schooling"])
        income = float(request.form["income"])

        # Create input DataFrame
        data = pd.DataFrame({
            "Life Expectancy at Birth (2021)": [life_expectancy],
            "Expected Years of Schooling (2021)": [schooling],
            "Gross National Income Per Capita (2021)": [income]
        })

        # Handle missing values
        data = imputer.transform(data)

        # Predict HDI
        prediction = model.predict(data)[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)