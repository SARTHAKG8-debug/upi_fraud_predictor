import os
from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

# Resolve paths relative to this file (works on Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model = joblib.load(os.path.join(BASE_DIR, "random_forest_model.pkl"))

# Load dataset to fit the LabelEncoder with the same data used during training
data = pd.read_csv(os.path.join(BASE_DIR, "upi_fraud_dataset.csv"))
label_encoder = LabelEncoder()
label_encoder.fit(data['upi_number'].astype(str))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form
        upi_raw = form_data['upi_number']

        # Encode UPI number (use -1 for unseen values)
        if upi_raw in label_encoder.classes_:
            upi_encoded = label_encoder.transform([upi_raw])[0]
        else:
            upi_encoded = -1

        input_data = {
            "trans_hour": int(form_data['trans_hour']),
            "trans_day": int(form_data['trans_day']),
            "trans_month": int(form_data['trans_month']),
            "trans_year": int(form_data['trans_year']),
            "trans_amount": float(form_data['trans_amount']),
            "upi_number": upi_encoded
        }

        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]
        result = "Fraud" if prediction == 1 else "Not Fraud"
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
