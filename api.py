from flask import Flask, request, jsonify, send_from_directory
import joblib, os, pandas as pd

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.abspath(__file__))

lr_model = joblib.load(os.path.join(BASE, 'logistic_regression_model.pkl'))
rf_model = joblib.load(os.path.join(BASE, 'random_forest_model.pkl'))
scaler   = joblib.load(os.path.join(BASE, 'scaler.pkl'))
encoders = joblib.load(os.path.join(BASE, 'label_encoders.pkl'))

CATEGORICAL = ["Gender", "Sleep Duration", "Dietary Habits"]
NUMERICAL   = ["Age", "Academic Pressure", "CGPA", "Study Satisfaction", "Work/Study Hours", "Financial Stress"]
FEATURES    = ["Gender", "Age", "Academic Pressure", "CGPA", "Study Satisfaction",
               "Sleep Duration", "Dietary Habits", "Work/Study Hours", "Financial Stress"]

@app.route('/')
def index():
    return send_from_directory(BASE, 'dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_type = data.get('model', 'lr')

    df = pd.DataFrame([{
        "Gender":             data['gender'],
        "Age":                float(data['age']),
        "Academic Pressure":  int(data['academic_pressure']),
        "CGPA":               float(data['cgpa']),
        "Study Satisfaction": int(data['study_satisfaction']),
        "Sleep Duration":     data['sleep_duration'],
        "Dietary Habits":     data['dietary_habits'],
        "Work/Study Hours":   float(data['work_study_hours']),
        "Financial Stress":   int(data['financial_stress']),
    }])

    for col in CATEGORICAL:
        df[col] = encoders[col].transform(df[col])

    df = df[FEATURES]
    df[NUMERICAL] = scaler.transform(df[NUMERICAL])

    model  = lr_model if model_type == 'lr' else rf_model
    label  = int(model.predict(df)[0])
    probas = model.predict_proba(df)[0]
    conf   = float(probas[label]) * 100

    return jsonify({
        'label':               label,
        'label_text':          'Distressed' if label == 1 else 'Not Distressed',
        'confidence':          round(conf, 2),
        'prob_distressed':     round(float(probas[1]) * 100, 2),
        'prob_not_distressed': round(float(probas[0]) * 100, 2),
        'model': 'Logistic Regression' if model_type == 'lr' else 'Random Forest'
    })

if __name__ == '__main__':
    print('StudentAI Dashboard running at http://localhost:8000')
    app.run(port=8000, debug=False)
