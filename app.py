import streamlit as st
import openai
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time
import streamlit_echarts as st_echarts
import warnings

warnings.filterwarnings('ignore')


st.markdown("## **🏋️ Personal Fitness Tracker**")


st.write("In this WebApp you will be able to observe your predicted calories burned in your body. ")

st.sidebar.header("User Input Parameters: ")

def user_input_features():
    st.sidebar.markdown("### Adjust your parameters below:")
    age = st.sidebar.slider("Age:", 10, 100, 30)
    bmi = st.sidebar.slider("BMI:", 15, 40, 20)
    duration = st.sidebar.slider("Workout Duration (min):", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate (bpm):", 60, 190, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C):", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender:", ("Male", "Female"))
    gender_male = 1 if gender_button == "Male" else 0

    # Use column names to match the training data
    data_model = {
    "Age": age,
    "BMI": bmi,
    "Duration": duration,
    "Heart_Rate": heart_rate,
    "Body_Temp": body_temp,
    "Gender_male": gender_male # Gender is encoded as 1 for male, o for female
    }
    features=pd.DataFrame(data_model,index=[0])
    return features

def calculate_heart_rate_zones(age):
    max_hr = 220 - age  # Maximum Heart Rate
    zones = {
        "Resting": (0, 0.5 * max_hr),
        "Warm-up": (0.5 * max_hr, 0.6 * max_hr),
        "Fat Burn": (0.6 * max_hr, 0.7 * max_hr),
        "Cardio": (0.7 * max_hr, 0.85 * max_hr),
        "Peak": (0.85 * max_hr, max_hr),
    }
    return zones


def heart_rate_gauge(heart_rate, age):
    zones = calculate_heart_rate_zones(age)
    options = {
        "series": [
            {
                "type": "gauge",
                "axisLine": {
                    "lineStyle": {
                        "width": 10,
                        "color": [
                            [0.5, "#32CD32"],  # Green - Resting
                            [0.6, "#FFD700"],  # Yellow - Warm-up
                            [0.7, "#FFA500"],  # Orange - Fat Burn
                            [0.85, "#FF4500"], # Red - Cardio
                            [1.0, "#8B0000"],  # Dark Red - Peak
                        ],
                    }
                },
                "pointer": {"width": 5},
                "detail": {"formatter": "{value} BPM"},
                "data": [{"value": heart_rate, "name": "Heart Rate"}],
                "min": 0,
                "max": zones["Peak"][1],  # Set max value dynamically
            }
        ]
    }
    st_echarts.st_echarts(options=options, height="300px")

def calculate_fatigue_score(data_model):
    """
    Estimate fatigue level based on heart rate trends, body temperature changes, and workout duration.
    
    Parameters:
    - data_model (dict): Dictionary containing user metrics including heart rate, body temp, and workout duration.
    
    Returns:
    - fatigue_score (int): A score from 1 to 10 indicating fatigue level.
    - recovery_time (str): Suggested recovery time.
    """
    
    # Extract data from model
    heart_rate = data_model["Heart_Rate"]
    body_temp = data_model["Body_Temp"]
    workout_duration = data_model["Duration"]
    
    # Calculate heart rate variability (assuming heart rate represents an average reading)
    hr_variability = min(heart_rate / 10, 5)  # Scale to a max of 5 points
    
    # Estimate temperature factor
    temp_factor = min((body_temp - 36.5) * 2, 3)  # Scale to a max of 3 points
    
    # Scale duration factor
    duration_factor = min(workout_duration / 20, 4)  # Scale to a max of 4 points
    
    # Compute fatigue score (out of 10)
    fatigue_score = min(int(hr_variability + temp_factor + duration_factor), 10)
    
    # Recommend recovery time based on fatigue score
    if fatigue_score <= 3:
        recovery_time = "Low fatigue - 12 hours rest recommended."
    elif fatigue_score <= 6:
        recovery_time = "Moderate fatigue - 24 hours rest recommended."
    else:
        recovery_time = "High fatigue - 48 hours rest recommended."
    
    return fatigue_score, recovery_time

def calculate_fatigue_score(heart_rate, body_temp, workout_duration):
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        bar.progress(i + 1)
        time.sleep(0.01)
    fatigue_score = (heart_rate * 0.3) + ((body_temp - 36) * 10) + (workout_duration * 0.2)
    return min(max(round(fatigue_score, 2), 1), 100)  # Ensure score is between 1 and 100

def recommend_recovery_time(fatigue_score):
    if fatigue_score < 30:
        return "Minimal fatigue detected. Recovery time: 4-8 hours. Light stretching and hydration recommended."
    elif fatigue_score < 60:
        return "Moderate fatigue detected. Recovery time: 12-24 hours. Consider a rest day or light activity."
    else:
        return "High fatigue detected! Recovery time: 24-48 hours. Prioritize sleep, hydration, and proper nutrition."


df = user_input_features()
st.markdown(
    """
    <style>
        div[data-testid="stProgress"] > div > div > div {
            background-color: #FFFFFF !important; /* Change this color */
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("---")
st.markdown("### 📌 Your Parameters")

latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)



# Personalized Workout Suggestions
st.markdown("---")
st.markdown("## 🏋️‍♂️ Personalized Workout Suggestions")

age = df['Age'][0]
bmi = df['BMI'][0]
duration = df['Duration'][0]

if bmi < 18.5:
    st.markdown("**🔹 Underweight - Focus on Strength Training & Healthy Diet**")
    st.markdown("- Strength Training: Squats, Push-ups, Deadlifts (3x a week)")
    st.markdown("- Eat more protein & healthy fats for muscle gain")
elif 18.5 <= bmi < 24.9:
    st.markdown("**✔️ Normal BMI - Maintain Fitness with a Balanced Routine**")
    st.markdown("- Cardio: Running, Cycling (20-30 min, 3x a week)")
    st.markdown("- Strength: Core Workouts (Planks, Russian Twists, etc.)")
elif 25 <= bmi < 29.9:
    st.markdown("**⚠️ Overweight - Focus on Weight Loss & Fat Burn**")
    st.markdown("- High-Intensity Interval Training: Jump Rope, Burpees, Mountain Climbers (15-20 min/day)")
    st.markdown("- Low-Impact Cardio: Brisk Walking, Swimming")
else:
    st.markdown("**🚨 Obese - Start with Light Exercise & Build Stamina**")
    st.markdown("- Start with: Walking 30 min/day, Yoga, Low-intensity Aerobics")
    st.markdown("- Progress to: Resistance Band Workouts & Light Strength Training")

if duration < 10:
    st.markdown("*⏳ Try increasing workout duration for better results!*")
elif duration > 30:
    st.markdown("🔥 *Great job on maintaining a high workout duration!* Keep pushing!")

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] /((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Separate features and labels
x_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

x_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(x_train, y_train)

# Align prediction data columns with training data
df = df.reindex(columns=x_train.columns, fill_value=0)

# Make prediction
prediction = random_reg.predict(df)
st.markdown(
    """
    <style>
        div[data-testid="stProgress"] > div > div > div {
            background-color: #FFFFFF !important; /* Change this color */
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("---")
st.markdown("### 🔥 Predicted Calories Burned")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

st.write(f"{round(prediction[0], 2)} ** KiloCalories ** ")

st.markdown(
    """
    <style>
        div[data-testid="stProgress"] > div > div > div {
            background-color: #FFFFFF !important; /* Change this color */
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("---")
st.markdown("### 📊 Similar Results")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

# Find similar results based on predicted calories
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write(similar_data.sample(5))

st.write(" --- ")
st.markdown("### ❤️ Heart Rate Zone Tracker")
heart_rate_gauge(int(df["Heart_Rate"][0]), int(df["Age"][0]))


st.markdown("#### 🔍 Insights:")
if df["Heart_Rate"][0] < calculate_heart_rate_zones(df["Age"][0])["Warm-up"][0]:
    st.markdown("✅ Your heart rate is in the **Resting Zone** – consider warming up.")
elif df["Heart_Rate"][0] < calculate_heart_rate_zones(df["Age"][0])["Fat Burn"][0]:
    st.markdown("🔥 You're in the **Warm-up Zone** – great for starting a session.")
elif df["Heart_Rate"][0] < calculate_heart_rate_zones(df["Age"][0])["Cardio"][0]:
    st.markdown("💪 You're in the **Fat Burn Zone** – ideal for weight loss & endurance.")
elif df["Heart_Rate"][0] < calculate_heart_rate_zones(df["Age"][0])["Peak"][0]:
    st.markdown("🚀 You're in the **Cardio Zone** – improving heart & lung capacity!")
else:
    st.markdown("⚠️ You're in the **Peak Zone** – be mindful of overexertion!")


# Hydration Tracker Section
st.markdown("---")
st.markdown("##💧 Hydration Tracker")

# Hydration formula estimates
duration = df["Duration"].values[0]  # Workout duration in minutes
heart_rate = df["Heart_Rate"].values[0]  # Heart rate in BPM
body_temp = df["Body_Temp"].values[0]  # Body temperature in Celsius

# Estimate fluid loss (approx. 0.5 - 1 L per hour)
fluid_loss = (0.5 + ((heart_rate - 60) / 130) * 0.5) * (duration / 60)  # Liters lost

# Adjust for body temperature impact (higher temp = more sweat)
if body_temp > 38:
    fluid_loss *= 1.2
elif body_temp < 36.5:
    fluid_loss *= 0.9

# Recommended water intake = 1.5x fluid loss to rehydrate properly
recommended_water = round(fluid_loss * 1.5, 2)

st.markdown(f"💦 **Estimated Fluid Loss:** {round(fluid_loss, 2)} L")
st.markdown(f"🥤 **Recommended Water Intake:** {recommended_water} L")

# Hydration Tips Based on Loss
if recommended_water < 0.5:
    st.markdown("✅ **Minimal dehydration risk** – A glass of water should suffice.")
elif 0.5 <= recommended_water < 1.5:
    st.markdown("⚠️ **Moderate fluid loss** – Drink at least **2-3 glasses of water**.")
else:
    st.markdown("🚨 **High fluid loss detected!** – Consider electrolyte drinks or **1L+ of water**.")





st.write(" --- ")
# Define meal plans
st.markdown(
    """
    <style>
        div[data-testid="stProgress"] > div > div > div {
            background-color: #FFFFFF !important; /* Change this color */
        }
    </style>
    """,
    unsafe_allow_html=True
)


meal_plans = {
    "low": {
        "name": "High-Protein Meal",
        "description": "Ideal for muscle retention and recovery.",
        "foods": ["Chicken", " Boiled eggs", "Yogurt", "Pasta"],
        "macros": {"Protein": "40g", "Carbs": "10g", "Fats": "5g"},
        "image": "https://www.pomelowines.com/wp-content/uploads/2021/02/PomeloRecipes-18-e1614377029820-1024x1024.jpg"
    },
    "moderate": {
        "name": "Balanced Energy Meal",
        "description": "Perfect mix of protein and carbs for energy.",
        "foods": ["Brown Rice", "Fish", "Veggies", "Salad"],
        "macros": {"Protein": "30g", "Carbs": "50g", "Fats": "10g"},
        "image": "https://i0.wp.com/wanderingchickpea.com/wp-content/uploads/2021/05/2370FBA9-03B1-4F87-862D-98A621C4049F.jpeg?resize=1062%2C1536&ssl=1"
    },
    "high": {
        "name": "Recovery Meal",
        "description": "Carb-rich meal for post-workout replenishment.",
        "foods": ["Dal Khichdi", "Oatmeal.", " Boiled egg", "Watermelon"],
        "macros": {"Protein": "35g", "Carbs": "70g", "Fats": "15g"},
        "image": "https://evolvedgenetics.com/wp-content/uploads/2023/03/Dal-Khichdi.jpg"
    }
}

st.write(" --- ")
st.markdown(" ### Fatigue & Recovery Score")

if st.button("Calculate Fatigue Score"):
    fatigue_score = calculate_fatigue_score(heart_rate,body_temp,duration)
    recovery_advice = recommend_recovery_time(fatigue_score)
    
    st.write(f"**Predicted Fatigue Score:** {fatigue_score}/100")
    st.write(f"**Recovery Recommendation:** {recovery_advice}")
st.write(" --- ")
# Get predicted calories
burned_calories = round(prediction[0], 2)
st.markdown("### 🍽️ Personalized Nutrition Plan")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)


# Select meal plan based on calories burned
if burned_calories < 150:
    meal_plan = meal_plans["low"]
elif 150 <= burned_calories <= 400:
    meal_plan = meal_plans["moderate"]
else:
    meal_plan = meal_plans["high"]



# Display meal plan
st.markdown(f"#### 🍲 {meal_plan['name']}")
st.write(meal_plan["description"])
st.image(meal_plan["image"], width=400)
st.write(f"**Recommended Foods:** {', '.join(meal_plan['foods'])}")

# Display macros
st.markdown("**📊 Macros Breakdown:**")
st.write(f"**Protein:** {meal_plan['macros']['Protein']} | **Carbs:** {meal_plan['macros']['Carbs']} | **Fats:** {meal_plan['macros']['Fats']}")




st.write(" --- ")
st.header("General Information: ")

# Boolean logic for age, duration, etc., compared to the user's input
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

# General Information
st.markdown("---")
st.markdown("### ℹ️ General Insights")
st.markdown("""
    - **👴 Age Comparison:** You are older than **{:.2f}%** of people.
    - **⏳ Workout Duration:** Your exercise duration is longer than **{:.2f}%** of people.
    - **💓 Heart Rate:** Your heart rate is higher than **{:.2f}%** of people.
    - **🌡️ Body Temperature:** Your body temperature is higher than **{:.2f}%** of people during exercise.
""".format(
    round(sum(boolean_age) / len(boolean_age) * 100, 2),
    round(sum(boolean_duration) / len(boolean_duration) * 100, 2),
    round(sum(boolean_heart_rate) / len(boolean_heart_rate) * 100, 2),
    round(sum(boolean_body_temp) / len(boolean_body_temp) * 100, 2)),
    unsafe_allow_html=True
)




st.markdown("---")

st.markdown("### 📊 Calories Burned vs Workout Duration")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
option = {
    "tooltip": {"trigger": "axis"},
    "xAxis": {
        "type": "category",
        "name": "Workout Duration (Minutes)",
        "nameLocation": "middle",
        "nameGap": 30,
        "data": list(range(5, df["Duration"].values[0] + 5, 5))
    },
    "yAxis": {
        "type": "value",
        "name": "Calories Burned",
        "nameLocation": "middle",
        "nameRotate": 90,
        "nameGap": 40
    },
    "series": [
        {
            "data": [round(prediction[0] * (i / df["Duration"].values[0]), 2) for i in range(5, df["Duration"].values[0] + 5, 5)],
            "type": "line",
            "smooth": True,
            "lineStyle": {"width": 3},
            "markPoint": {
                "data": [
                    {"type": "max", "name": "Max Calories Burned"},
                    {"type": "min", "name": "Min Calories Burned"}
                ]
            }
        }
    ]
}
st_echarts.st_echarts(options=option)


st.markdown("---")
# 🔍 **Graph Insights & Takeaways**
st.markdown("### 📢 What This Graph Tells You")

# ✅ If short workout duration
if df["Duration"].values[0] < 15:
    st.markdown("""
    🚀 **Your workout duration is quite short.**
    - Increasing your workout time will significantly boost calorie burn.  
    - Aim for at least **20–30 minutes** for better results.  
    - Try **adding cardio to improve efficiency.  
    """)

# ✅ If workout duration is balanced (15-30 min)
elif 15 <= df["Duration"].values[0] <= 30:
    st.markdown("""
    🎯 **You have a solid workout duration!**
    - Your calorie burn increases consistently with time.  
    - For **more fat burn**, try increasing intensity instead of duration.  
    - Consider **interval training or resistance workouts**.  
    """)

# ✅ If workout is very long (>30 min)
else:
    st.markdown("""
    🔥 **You're working out for a long duration!**
    - Longer workouts lead to **higher calorie burn**, but watch out for fatigue.  
    - If energy levels drop, consider **shorter high-intensity workouts** instead.  
    - Make sure to **stay hydrated and fuel your body properly**.  
    """)


st.markdown("---")
st.markdown("""
###  Your Progress Summary:
✅ **Longer workouts burn more calories** – Keep pushing your limits!  
✅ **Your peak calorie burn is marked** so you can aim higher.  
✅ **Balance is key** – Too long without rest can lead to burnout.  
✅ **Next Steps**: Adjust workout intensity or duration to meet your fitness goals.  

Consistency is 🔑 to fitness success! Keep going! 💪🔥  
""")

