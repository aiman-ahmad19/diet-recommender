import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64

# ----------------- Streamlit Config -----------------
st.set_page_config(page_title="Personalized Diet Recommender", layout="wide")

# ----------------- Background & Fonts -----------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif !important;
    }}

    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
    }}

    .block-container {{
        background-color: rgba(255, 255, 255, 0.94);
        padding: 3rem 2rem;
        border-radius: 15px;
    }}

    h1, h2, h3, h4, h5, h6, p, label, span, div {{
        font-family: 'Poppins', sans-serif !important;
    }}

    .stButton>button {{
        font-size: 18px;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
    }}

    .bmi-box {{
        font-size: 22px;
        background-color: #eaf6ff;
        color: #003366;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-top: 10px;
        text-align: center;
        font-weight: bold;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

add_bg_from_local("background.png")

# ----------------- Header -----------------
st.title("🥗 Personalized Diet Recommendation System")
st.markdown("Get tailored meal plans based on your health, habits, and preferences.")

# ----------------- Load Models -----------------
model = joblib.load("models/diet_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# ----------------- Helpers -----------------
def safe_encode(label_encoder, value):
    if value == "None":
        return label_encoder.transform([label_encoder.classes_[0]])[0]
    if value in label_encoder.classes_:
        return label_encoder.transform([value])[0]
    return label_encoder.transform([label_encoder.classes_[0]])[0]

def get_options_with_none(label_encoder):
    return ['None'] + list(label_encoder.classes_)

# ----------------- Input Section -----------------
st.markdown("## 👤 Basic Information")
age = st.slider("Age", 10, 100, 25)
gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=65)
bmi = weight / ((height / 100) ** 2)
st.markdown(f'<div class="bmi-box">Body Mass Index (BMI): {bmi:.2f}</div>', unsafe_allow_html=True)

st.markdown("## 🏥 Health Conditions")
chronic = st.selectbox("Chronic Disease", get_options_with_none(label_encoders['Chronic_Disease']))
systolic = st.slider("Systolic BP", 80, 200, 120)
diastolic = st.slider("Diastolic BP", 50, 130, 80)
chol = st.selectbox("Cholesterol Level", label_encoders['Cholesterol_Level'].classes_)
sugar = st.slider("Blood Sugar Level", 60, 400, 100)
genetic = st.selectbox("Genetic Risk Factor", label_encoders['Genetic_Risk_Factor'].classes_)
allergies = st.selectbox("Allergies", get_options_with_none(label_encoders['Allergies']))

st.markdown("## 💪 Lifestyle & Habits")
steps = st.slider("Daily Steps", 0, 30000, 5000, step=500)
exercise = st.selectbox("Exercise Frequency", label_encoders['Exercise_Frequency'].classes_)
sleep = st.slider("Sleep Hours", 0, 24, 8)
alcohol = st.selectbox("Alcohol Consumption", label_encoders['Alcohol_Consumption'].classes_)
smoking = st.selectbox("Smoking Habit", label_encoders['Smoking_Habit'].classes_)
diet_habit = st.selectbox("Dietary Habits", label_encoders['Dietary_Habits'].classes_)

st.markdown("## 🍽️ Diet Preferences")
calories = st.slider("Daily Caloric Intake", 1000, 5000, 2200)
protein = st.slider("Protein Intake (g)", 0, 300, 70)
carbs = st.slider("Carbohydrate Intake (g)", 0, 500, 200)
fats = st.slider("Fat Intake (g)", 0, 200, 70)
cuisine = st.selectbox("Preferred Cuisine", label_encoders['Preferred_Cuisine'].classes_)
aversion = st.selectbox("Food Aversions", label_encoders['Food_Aversions'].classes_)

# ----------------- Data Preparation -----------------
input_data = {
    'Age': age,
    'Gender': safe_encode(label_encoders['Gender'], gender),
    'Height_cm': height,
    'Weight_kg': weight,
    'BMI': bmi,
    'Chronic_Disease': safe_encode(label_encoders['Chronic_Disease'], chronic),
    'Blood_Pressure_Systolic': systolic,
    'Blood_Pressure_Diastolic': diastolic,
    'Cholesterol_Level': safe_encode(label_encoders['Cholesterol_Level'], chol),
    'Blood_Sugar_Level': sugar,
    'Genetic_Risk_Factor': safe_encode(label_encoders['Genetic_Risk_Factor'], genetic),
    'Allergies': safe_encode(label_encoders['Allergies'], allergies),
    'Daily_Steps': steps,
    'Exercise_Frequency': safe_encode(label_encoders['Exercise_Frequency'], exercise),
    'Sleep_Hours': sleep,
    'Alcohol_Consumption': safe_encode(label_encoders['Alcohol_Consumption'], alcohol),
    'Smoking_Habit': safe_encode(label_encoders['Smoking_Habit'], smoking),
    'Dietary_Habits': safe_encode(label_encoders['Dietary_Habits'], diet_habit),
    'Caloric_Intake': calories,
    'Protein_Intake': protein,
    'Carbohydrate_Intake': carbs,
    'Fat_Intake': fats,
    'Preferred_Cuisine': safe_encode(label_encoders['Preferred_Cuisine'], cuisine),
    'Food_Aversions': safe_encode(label_encoders['Food_Aversions'], aversion)
}

input_df = pd.DataFrame([input_data])
scaled_input = scaler.transform(input_df)

# ----------------- Prediction -----------------
if st.button("🍽️ Get Diet Recommendation"):
    try:
        prediction = model.predict(scaled_input)[0]
        meal_plan_label = label_encoders['Recommended_Meal_Plan'].inverse_transform([prediction])[0]
        st.success(f"✅ Recommended Meal Plan: **{meal_plan_label}**")

        sample_meal_plans = {
            "Low-Fat Diet": """🍳 **Breakfast**: Oatmeal with fruits and skim milk  
🥗 **Lunch**: Grilled chicken salad  
🥘 **Dinner**: Steamed fish with rice and vegetables""",
            "High-Protein Diet": """🍳 **Breakfast**: Eggs with spinach  
🍗 **Lunch**: Chicken breast with quinoa  
🥩 **Dinner**: Salmon with lentils""",
            "Low-Carb Diet": """🥚 **Breakfast**: Eggs and avocado  
🥗 **Lunch**: Tuna salad  
🍖 **Dinner**: Grilled chicken with cauliflower""",
            "Balanced Diet": """🍽️ **Breakfast**: Yogurt with granola  
🍛 **Lunch**: Rice, chicken, vegetables  
🍲 **Dinner**: Fish, sweet potatoes, beans""",
            "Vegetarian Diet": """🍓 **Breakfast**: Smoothie with almond milk  
🥦 **Lunch**: Lentil curry  
🍝 **Dinner**: Veg stir-fry with tofu""",
            "Vegan Diet": """🥝 **Breakfast**: Chia pudding  
🌮 **Lunch**: Hummus wrap  
🍛 **Dinner**: Chickpea stew with couscous""",
            "Keto Diet": """🥑 **Breakfast**: Eggs with avocado  
🥓 **Lunch**: Bacon-wrapped chicken  
🍖 **Dinner**: Salmon with spinach""",
            "Diabetic Diet": """🍎 **Breakfast**: Whole grain toast with peanut butter  
🥗 **Lunch**: Turkey with greens  
🥘 **Dinner**: Lentil soup and barley""",
            "Mediterranean Diet": """🥥 **Breakfast**: Yogurt with honey  
🍴 **Lunch**: Fish with bulgur  
🍅 **Dinner**: Tomato soup with bread""",
            "Paleo Diet": """🥚 **Breakfast**: Omelette  
🥗 **Lunch**: Lettuce turkey wraps  
🍗 **Dinner**: Chicken with sweet potatoes""",
            "Gluten-Free Diet": """🍓 **Breakfast**: GF oatmeal  
🥗 **Lunch**: Quinoa salad  
🥘 **Dinner**: Rice noodles with tofu""",
            "Heart-Healthy Diet": """🍳 **Breakfast**: Whole grain cereal  
🥗 **Lunch**: Salmon salad  
🍲 **Dinner**: Chicken, rice, broccoli""",
            "Renal Diet": """🥣 **Breakfast**: Apple slices and almond butter  
🥗 **Lunch**: Rice with turkey  
🍛 **Dinner**: Couscous and roasted peppers""",
            "DASH Diet": """🍊 **Breakfast**: Yogurt with peach  
🥗 **Lunch**: Tuna sandwich  
🍲 **Dinner**: Chicken with sweet potato""",
            "Weight Loss Diet": """🥚 **Breakfast**: Eggs and grapefruit  
🥗 **Lunch**: Veggies with lean chicken  
🍛 **Dinner**: Clear soup and tofu""",
            "Weight Gain Diet": """🍞 **Breakfast**: PB toast and smoothie  
🍲 **Lunch**: Pasta with chicken  
🥘 **Dinner**: Rice, lentils, paneer"""
        }

        st.markdown("### 🧾 Sample Meal Plan")
        st.markdown(sample_meal_plans.get(meal_plan_label, "No meal plan available."))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
