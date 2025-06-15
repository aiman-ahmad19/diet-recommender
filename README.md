# diet-recommender
Personalized Diet Recommender
A health-focused web application that uses machine learning to recommend personalized diet plans based on an individual’s age, gender, health metrics, lifestyle habits, and dietary preferences. Built with Streamlit and powered by a trained classification model.

Features
-Tailored meal plan recommendations
-BMI calculation and health tracking
-Support for health conditions, allergies, and diet preferences
-Interactive UI with stylish background and clean layout
-Sample meal plans for different diet types

Folder Structure
diet_recommender/
│
├── app.py                      # Core Streamlit app
├── app_streamlit.py           # Alternate app version
├── main.py                    # Main script (if used in backend)
├── background.png             # Background image for UI
│
├── models/                    # ML artifacts
│   ├── diet_model.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
│
└── data/
    └── Personalized_Diet_Recommendations.csv
Dataset and Model Training
Dataset: The project uses a custom dataset (Personalized_Diet_Recommendations.csv) containing features such as:

-Age, Gender, Height, Weight, BMI
-Chronic diseases, Blood pressure, Sugar level
-Lifestyle info (sleep, exercise, steps, smoking, etc.)
-Dietary preferences, cuisine choices, and food aversions
-The target column is the Recommended Meal Plan

Preprocessing:
-Categorical features were encoded using LabelEncoder
-Numerical features were scaled using StandardScaler
-BMI was calculated dynamically from height and weight

Model Used:
-The final classification model is a Random Forest Classifier, chosen for its accuracy and robustness with mixed-type data.
-The model was trained, validated, and saved using joblib.

Graphs and Evaluation Curves
Confusion Matrix:
Visualizes true vs. predicted diet classes to measure classification accuracy.

Classification Report (Precision, Recall, F1-score):
A detailed textual report showing how well the model performs per class.

ROC Curve:
Shows the tradeoff between sensitivity and specificity for each class (One-vs-Rest approach).

Feature Importance:
Highlights which input features most influenced the model’s diet prediction decisions.

All evaluation metrics were computed on a test split to ensure generalization.

Tech Stack
-Python 3
-Streamlit — for frontend
-scikit-learn — model training and preprocessing
-pandas, numpy — data handling
-joblib — model serialization

How It Works
-User inputs health and lifestyle details
-Input data is preprocessed and scaled
-The trained classification model predicts the best diet plan
-A recommended diet and meal plan is shown in real-time

UI Preview
-Background image included
-Stylish layout and readable fonts
-Real-time BMI calculation

Author
Aiman Ahmad
BSDSF22A020

License
This project is for educational and non-commercial use. Modify freely but please credit the author.

