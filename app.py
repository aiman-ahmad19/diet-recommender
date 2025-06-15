from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, scaler, encoders
model = joblib.load('models/diet_classifier_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')

# Define expected feature order
features = ['Age', 'Gender', 'Weight', 'Height', 'BMI', 'Activity_Level', 'Health_Issues',
            'Dietary_Habits', 'Sleep_Hours', 'Caloric_Intake', 'Exercise_Frequency']

@app.route('/')
def index():
    return "Personalized Diet Recommender API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # Encode categorical features
    for col in label_encoders:
        if col in input_df.columns:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])

    # Reorder columns
    input_df = input_df[features]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    label = list(label_encoders['Recommended_Meal_Plan'].classes_)[prediction[0]]

    return jsonify({'recommended_meal_plan': label})

if __name__ == '__main__':
    app.run(debug=True)
