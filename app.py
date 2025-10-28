from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset for stats
df = pd.read_csv("CollegePlacement.csv")

# Pre-calculate placement stats
placed_count = df[df['Placement'] == 'Yes'].shape[0]
not_placed_count = df[df['Placement'] == 'No'].shape[0]
total = placed_count + not_placed_count
placement_rate = round((placed_count / total) * 100, 2)

@app.route('/')
def home():
    return render_template(
        "index.html",
        placed=placed_count,
        not_placed=not_placed_count,
        rate=placement_rate
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    IQ = float(request.form['IQ'])
    Prev_Sem_Result = float(request.form['Prev_Sem_Result'])
    CGPA = float(request.form['CGPA'])
    Academic_Performance = float(request.form['Academic_Performance'])
    Internship_Experience = 1 if request.form['Internship_Experience'] == 'Yes' else 0
    Extra_Curricular_Score = float(request.form['Extra_Curricular_Score'])
    Communication_Skills = float(request.form['Communication_Skills'])
    Projects_Completed = float(request.form['Projects_Completed'])
    
    # Prepare features
    features = np.array([[IQ, Prev_Sem_Result, CGPA, Academic_Performance,
                          Internship_Experience, Extra_Curricular_Score,
                          Communication_Skills, Projects_Completed]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    result = "Placed 🎉" if prediction == 1 else "Not Placed 😔"
    return render_template("result.html", prediction=result)
    

if __name__ == '__main__':
    app.run(debug=True)
