from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import os
import pandas as pd

app = Flask(__name__)

heart_model = joblib.load("models/heart_model.pkl")

stress_model = joblib.load("models/stress_model.pkl")



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        disease = data.get("disease")

        # ---------- HEART ----------
        if disease == "heart":
            X = np.array([[ 
                float(data["age"]),
                int(data["sex"]),
                int(data["cp"]),
                float(data["trestbps"]),
                float(data["chol"]),
                float(data["thalach"]),
                int(data["exang"])
            ]])

            pred = heart_model.predict(X)[0]
            prob = heart_model.predict_proba(X)[0][1] * 100
            result = "High Risk" if pred == 1 else "Low Risk"

            return jsonify({"result": f"{result} ({prob:.2f}%)"})

        # ---------- STRESS ----------
        elif disease == "stress":
            import pandas as pd

            X = pd.DataFrame([{
                "Age": float(data["age"]),
                "Sleep Duration": float(data["sleep_duration"]),
                "Quality of Sleep": float(data["sleep_quality"]),
                "Physical Activity Level": float(data.get("physical_activity", 0)),
                "Heart Rate": float(data["heart_rate"]),
                "Daily Steps": float(data["steps"])
            }])

            score = int(stress_model.predict(X)[0])

    # safety clamp (0–10)
            score = max(0, min(score, 10))

            return jsonify({
        "result": f"Stress Level Score: {score} / 10"})

        return jsonify({"error": "Invalid disease"}), 400

    except Exception as e:
        print("REAL ERROR:", e)   
        return jsonify({"error": "Prediction failed"}), 400



@app.route("/download-report", methods=["POST"])
def download_report():
    data = request.get_json()
    os.makedirs("reports", exist_ok=True)

    file_path = "reports/health_report.pdf"
    c = canvas.Canvas(file_path, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, h - 50, "Health Prediction Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, h - 90, f"Name: {data['name']}")
    c.drawString(50, h - 115, f"Disease: {data['disease'].title()}")
    c.drawString(50, h - 140, f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}")

    y = h - 180
    for k, v in data["inputs"].items():
        c.drawString(50, y, f"{k.replace('_',' ').title()}: {v}")
        y -= 22

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 10, f"Result: {data['result']}")

    c.save()
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
