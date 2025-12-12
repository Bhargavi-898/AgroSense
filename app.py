import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import requests
from datetime import datetime
from fpdf import FPDF
import io
from gtts import gTTS
import re
import bcrypt
from deep_translator import GoogleTranslator
from db import create_user, get_user_by_email, update_password, history_col
import os






import bcrypt
from db import create_user, get_user_by_email, update_password

GMAIL_PATTERN = r"^[a-zA-Z0-9._%+-]+@gmail\.com$"

def is_valid_gmail(email):
    return re.match(GMAIL_PATTERN, email) is not None


# Default session state users
if "users" not in st.session_state:
    st.session_state["users"] = {"admin": "admin123"}  # default user
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None


def login_page():
    st.title(t("🔐 Login to Agrosense"))

    menu = t(["Login", "Register", "Forgot Password"])
    choice = st.radio(t("Select Action"), menu)

    # ---------------- LOGIN ----------------
    if choice == t("Login"):
        with st.form("login_form"):
            email = st.text_input(t("Email"))
            password = st.text_input(t("Password"), type="password")
            submit = st.form_submit_button(t("Login"))

            if submit:

                # Gmail domain validation
                if not is_valid_gmail(email):
                    st.error(t("Email must end with @gmail.com"))
                    st.stop()

                user = get_user_by_email(email)

                if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = email
                    st.success(t(f"Welcome, {email}!"))
                    st.stop()
                else:
                    st.error(t("Invalid email or password"))

    # ---------------- REGISTER ----------------
    elif choice == t("Register"):
        with st.form("register_form"):
            new_email = st.text_input(t("Enter Email"))
            new_pass = st.text_input(t("Create Password"), type="password")
            submit = st.form_submit_button(t("Register"))

            if submit:

                if not new_email or not new_pass:
                    st.warning(t("Email and password required"))
                    st.stop()

                # Gmail validation
                if not is_valid_gmail(new_email):
                    st.error(t("Email must end with @gmail.com"))
                    st.stop()

                hashed = bcrypt.hashpw(new_pass.encode("utf-8"), bcrypt.gensalt())
                ok, msg = create_user(new_email, hashed)

                if ok:
                    st.success(t("Registration successful! Please login."))
                else:
                    st.warning(msg)

    # ---------------- FORGOT PASSWORD ----------------
    elif choice == t("Forgot Password"):
        with st.form("forgot_form"):
            reset_email = st.text_input(t("Enter your registered email"))
            new_pass = st.text_input(t("New Password"), type="password")
            confirm_pass = st.text_input(t("Confirm New Password"), type="password")
            submit = st.form_submit_button(t("Reset Password"))

            if submit:

                if not reset_email or not new_pass or not confirm_pass:
                    st.warning(t("All fields required"))
                    st.stop()

                # Gmail validation
                if not is_valid_gmail(reset_email):
                    st.error(t("Email must end with @gmail.com"))
                    st.stop()

                if new_pass != confirm_pass:
                    st.error(t("Passwords do not match"))
                    st.stop()

                hashed = bcrypt.hashpw(new_pass.encode("utf-8"), bcrypt.gensalt())
                ok, msg = update_password(reset_email, hashed)

                if ok:
                    st.success(t("Password reset successful! Please login again."))
                else:
                    st.error(msg)




# ================== END LOGIN SYSTEM ==================


# 🌐 Language Mapping
lang_map = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi"
}

# Sidebar for language selection
language = st.sidebar.selectbox("🌐 Select Language", options=list(lang_map.keys()), index=0)
lang_code = lang_map[language]   # ← store dynamic language code

# Translation function
def t(text):
    """Translate UI text to selected language"""
    try:
        return GoogleTranslator(source="auto", target=lang_code).translate(text)
    except:
        return text   # fallback if translation fails


def speak(text):
    tts = gTTS(text=text, lang=lang_code)
    tts.save("voice.mp3")
    audio_file = open("voice.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")
def clean_pdf_text(text):
    # Replace long dash with hyphen
    text = text.replace("—", "-")
    # Remove emojis and non-ASCII characters
    return re.sub(r"[^\x00-\x7F]+", "", text)

# App Config
st.set_page_config(page_title="Agrosense - Smart Crop Recommendation", layout="centered")
st.title("🌾 Agrosense - Smart Crop Recommendation System")

# Load data
df = pd.read_csv("Crop_recommendation.csv")
yield_df = pd.read_csv("yield_price_data.csv")


X = df.drop('label', axis=1)
y = df['label']
model = RandomForestClassifier()
model.fit(X, y)

# --- helper: convert raw list to DataFrame with the same column names used for training ---
def make_input_df(values):
    """
    Converts a list or 1D array of feature values into a DataFrame
    with the same column names and dtypes as the training data X.

    Parameters:
        values (list or 1D array): Feature values in the same order as X.columns

    Returns:
        pd.DataFrame: DataFrame with one row, ready for prediction
    """
    # Ensure it's a list
    vals = list(values)

    # Create DataFrame with same columns as training data
    df_in = pd.DataFrame([vals], columns=X.columns)

    # Cast each column to match training data dtype
    for col in df_in.columns:
        try:
            df_in[col] = df_in[col].astype(X[col].dtype)
        except Exception:
            # ignore if conversion fails
            pass

    return df_in


fertilizer_data = {
    "rice": "N: 90, P: 40, K: 40 — Apply in 3 splits (basal, tillering, panicle).",
    "wheat": "N: 120, P: 60, K: 40 — Split into basal and top-dressing doses.",
    "maize": "N: 150, P: 75, K: 40 — Apply 1/3 at sowing, rest in two splits.",
    "cotton": "N: 75, P: 30, K: 30 — Apply half during sowing, rest at flowering.",
    "ground nut": "N: 20, P: 40, K: 40 — Apply all at sowing.",
    "gram": "N: 20, P: 40, K: 0 — Apply all before sowing.",
    "peas": "N: 25, P: 50, K: 25 — Use Rhizobium inoculation + basal dose.",
    "mustard": "N: 80, P: 40, K: 40 — Apply full dose before sowing.",
    "watermelon": "N: 100, P: 50, K: 50 — Basal + top-dress in 2 weeks.",
    "muskmelon": "N: 80, P: 40, K: 40 — Split over 3 stages.",
    "cucumber": "N: 70, P: 35, K: 35 — Balanced NPK with organic manure.",
    "jute": "N: 60, P: 30, K: 30 — Apply full dose before sowing.",
    "banana": "N: 200, P: 60, K: 200 — Split over 4 applications yearly.",
    "apple": "N: 400, P: 250, K: 300 — Applied annually based on tree age.",
    "grapes": "N: 160, P: 100, K: 150 — Before pruning and after fruit set.",
    "coffee": "N: 120, P: 90, K: 120 — Split into 2–3 applications yearly.",
    "papaya": "N: 200, P: 150, K: 200 — Monthly split applications recommended.",
    "orange": "N: 300, P: 150, K: 250 — Annual dose split into 2 parts.",
    "mango": "N: 250, P: 200, K: 300 — After harvest and before flowering.",
    "pomegranate": "N: 150, P: 100, K: 150 — After pruning and flowering.",
    "blackgram": "N: 20, P: 40, K: 20 — Apply all at sowing.",
    "mungbean": "N: 20, P: 40, K: 20 — Apply full basal dose.",
    "lentil": "N: 25, P: 50, K: 25 — Rhizobium seed treatment recommended.",
    "pigeonpeas": "N: 25, P: 50, K: 25 — Basal application before sowing.",
    "kidneybeans": "N: 30, P: 60, K: 30 — Use well-decomposed FYM also.",
    "mothbeans": "N: 20, P: 40, K: 20 — Apply entire dose before sowing.",
    "sunflower": "N: 60, P: 60, K: 40 — Full dose before sowing.",
    "soybean": "N: 30, P: 60, K: 30 — Use Rhizobium culture and apply basal.",
    "sorghum": "N: 100, P: 50, K: 40 — Split into 2 applications."
}

# Crop Rotation Knowledge Graph
rotation_rules = {
    "rice": "Chickpea or Mustard",
    "wheat": "Blackgram or Groundnut",
    "maize": "Pigeonpeas or Lentil",
    "cotton": "Mungbean or Chickpea",
    "ground nut": "Maize or Sorghum",
    "gram": "Wheat or Rice",
    "peas": "Maize or Mustard",
    "mustard": "Green Gram or Soybean",
    "watermelon": "Chickpea or Maize",
    "muskmelon": "Pigeonpeas or Green Gram",
    "cucumber": "Wheat or Chickpea",
    "jute": "Paddy or Wheat",
    "banana": "Pulses (Blackgram, Chickpea)",
    "apple": "Grasses or Cover Crops",
    "grapes": "Cover Crops (Legumes)",
    "coffee": "Pepper or Banana",
    "papaya": "Green Gram",
    "orange": "Legume Cover Crops",
    "mango": "Short Duration Pulses",
    "pomegranate": "Groundnut or Lentil",
    "blackgram": "Rice or Maize",
    "mungbean": "Wheat or Sorghum",
    "lentil": "Maize or Cotton",
    "pigeonpeas": "Wheat or Mustard",
    "kidneybeans": "Wheat or Rice",
    "mothbeans": "Maize or Chickpea",
}
# ✅ Move these BEFORE they're used
rotation_cycle = {
    "legume": {
        "description": "Nitrogen fixing crops that improve soil fertility.",
        "crops": [
            "gram",
            "peas",
            "black gram",
            "mung bean",
            "lentil",
            "pigeon peas",
            "kidney beans",
            "moth beans",
            "cowpea",
            "soybean",
            "alfalfa",
            "green gram",
            "broad beans",
            "french beans",
            "cluster beans"
        ]
    },

    "cereal": {
        "description": "Heavy nutrient-consuming crops (heavy feeders).",
        "crops": [
            "rice",
            "wheat",
            "maize",
            "sorghum",
            "millet",
            "barley",
            "oats",
            "rye",
            "triticale"
        ]
    },

    "oilseed": {
        "description": "Deep-rooted crops that improve soil structure.",
        "crops": [
            "mustard",
            "groundnut",
            "sunflower",
            "soybean",
            "sesame",
            "rapeseed",
            "linseed",
            "safflower",
            "castor"
        ]
    },

    "root_crops": {
        "description": "Deep-root crops that help break soil compaction.",
        "crops": [
            "carrot",
            "radish",
            "turnip",
            "beetroot",
            "sweet potato",
            "tapioca",
            "cassava",
            "yam",
            "potato"
        ]
    },

    "vegetables": {
        "description": "Short-duration crops; good in intermediate cycle.",
        "fruit_vegetables": [
            "tomato",
            "brinjal",
            "chilli",
            "pumpkin",
            "bottle gourd",
            "bitter gourd",
            "cucumber"
        ],
        "leafy_vegetables": [
            "spinach",
            "fenugreek",
            "lettuce",
            "coriander"
        ],
        "flower_vegetables": [
            "cauliflower",
            "cabbage",
            "broccoli"
        ]
    },

    "fodder_crops": {
        "description": "Soil-restorative crops grown to revive soil.",
        "crops": [
            "clover",
            "berseem",
            "alfalfa",
            "sudan grass",
            "cowpea fodder",
            "green manure crops",
            "ryegrass"
        ]
    },

    "cash_crops": {
        "description": "Long-duration crops; placed at start of a cycle.",
        "crops": [
            "sugarcane",
            "cotton",
            "tobacco",
            "banana"
        ]
    }
}


crop_type_map = {}
for crop_type, crop_list in rotation_cycle.items():
    for crop in crop_list:
        crop_type_map[crop.lower()] = crop_type
# Weather API
API_KEY = "b982d2b22599560cd7eff5c6815d0159"

def get_weather(city):
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": API_KEY, "units": "metric"}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            temp = data["main"]["temp"]
            hum = data["main"]["humidity"]
            rain = data.get("rain", {}).get("1h", 0.0) or 0.0
            return temp, hum, rain
        else:
            st.error("API Error: " + response.json().get("message", "Unknown"))
            return None, None, None
    except Exception as e:
        st.error(f"Weather fetch error: {e}")
        return None, None, None

if not st.session_state["logged_in"]:
    login_page()
    st.stop()  # stop execution until logged in
else:
    st.sidebar.write(f"👋 Logged in as: {st.session_state['current_user']}")
    if st.sidebar.button("🚪 Logout"):
        st.session_state["logged_in"] = False
        st.session_state["current_user"] = None
        st.stop()

    # Sidebar Navigation
    page = st.sidebar.radio("Navigate", [
        "🏠 Home", 
        "🌦️ Input Data", 
        "🌱 Recommend Crop", 
        "🧪 Fertilizer Suggestion", 
        "📊 Historical Weather",
        "📈 Yield & Forecast", 
        "♻️ Crop Rotation Plan",
        "📊 Crop Prediction Insights",
        "📄 Download Report"
    ])


# Home
if page == "🏠 Home":
    st.header(t("Welcome to Agrosense 👋"))
    st.markdown(t("""
    Agrosense helps farmers and agriculturists make informed crop choices using:
    - ✅ Soil Nutrient Data (N, P, K, pH)
    - ✅ Real-Time Weather (Temperature, Humidity, Rainfall)
    - ✅ Seasonal Suitability Checks
    - ✅ Fertilizer Suggestions
    - ✅ PDF Recommendations
    """))


# Input
elif page == "🌦️ Input Data":
    st.header(t("🧪 Enter Soil & Weather Conditions"))

    N = st.slider(t("Nitrogen (N)"), 0, 140, 90)
    P = st.slider(t("Phosphorus (P)"), 5, 145, 42)
    K = st.slider(t("Potassium (K)"), 5, 205, 43)
    ph = st.slider(t("pH Level"), 3.5, 9.5, 6.5)

    city = st.text_input(t("🌍 Enter City for Live Weather"), "Vijayawada")
    st.session_state["city"] = city

    if st.button(t("📡 Fetch Weather")):
        temp, hum, rain = get_weather(city)
        if temp is not None:
            st.success(t(f"Weather in {city}"))
            st.write(t(f"🌡️ Temperature: {temp} °C"))
            st.write(t(f"💧 Humidity: {hum} %"))
            st.write(t(f"🌧️ Rainfall: {rain} mm"))
            st.session_state.temp = temp
            st.session_state.hum = hum
            st.session_state.rain = rain
        else:
            st.error(t("Could not fetch weather."))
    else:
        temp = st.slider(t("Temperature (°C)"), 8.0, 45.0, 25.0)
        hum = st.slider(t("Humidity (%)"), 10.0, 100.0, 80.0)
        rain = st.slider(t("Rainfall (mm)"), 20.0, 300.0, 100.0)

    st.session_state.update({
        'N': N, 'P': P, 'K': K, 'ph': ph,
        'temperature': st.session_state.get("temp", temp),
        'humidity': st.session_state.get("hum", hum),
        'rainfall': st.session_state.get("rain", rain),
    })

    
# Recommend Crop
elif page == "🌱 Recommend Crop":
    st.header(t("🌱 Recommended Crop Based on Conditions"))

    if 'N' in st.session_state:
        input_data = [
            st.session_state['N'],
            st.session_state['P'],
            st.session_state['K'],
            st.session_state['temperature'],
            st.session_state['humidity'],
            st.session_state['ph'],
            st.session_state['rainfall']
        ]

        # --- Convert input list to DataFrame with column names ---
        input_df = make_input_df(input_data)
        probabilities = model.predict_proba(input_df)[0]  # now no warning
        top_indices = np.argsort(probabilities)[::-1][:3]
        crops = model.classes_
        recommended_crop = crops[top_indices[0]]
        st.session_state["recommendation"] = recommended_crop

        # Store top 3 crops for fertilizer suggestion dropdown
        top_3_crops = [crops[i] for i in top_indices]
        st.session_state["fertilizer_recommendations"] = top_3_crops

        st.success(t(f"✅ Best Crop: **{recommended_crop}**"))

        # Show Top 3 Crops Pie Chart
        st.write(t("### 🥈 Other Good Options:"))
        for i in top_indices[1:]:   
            st.write(t(f"- {crops[i]} ({round(probabilities[i]*100, 2)}%)"))

        # Determine Current Season
        month = datetime.now().month
        season = t("Kharif") if 6 <= month <= 9 else t("Rabi") if 10 <= month <= 3 else t("Zaid")
        st.info(t(f"📅 Current Season: **{season}**"))

        

        
elif page == "🧪 Fertilizer Suggestion":
    st.header(t("🧪 Fertilizer Suggestion Based on Your Input"))

    # Check if input and recommendation data exists
    if 'N' not in st.session_state or 'fertilizer_recommendations' not in st.session_state:
        st.warning(t("⚠️ Please complete input and crop recommendation first."))
    else:
        top_crops = st.session_state['fertilizer_recommendations']  # list of top 3 crops
        selected_crop = st.selectbox(t("🌾 Select a crop from recommended list"), top_crops)

    if st.button(t("📊 Show Fertilizer Recommendation")):
        # Get your input values
        N_input = st.session_state["N"]
        P_input = st.session_state["P"]
        K_input = st.session_state["K"]

        import re
        rec = fertilizer_data.get(selected_crop.lower(), "")
        match = re.findall(r"N:\s*(\d+),\s*P:\s*(\d+),\s*K:\s*(\d+)", rec)
        
        if match:
            rec_n, rec_p, rec_k = map(int, match[0])
            st.write(t(f"**Recommended NPK for {selected_crop.capitalize()}:** N: {rec_n}, P: {rec_p}, K: {rec_k} — {rec.split('—')[-1].strip()}"))

            # Nitrogen
            if abs(N_input - rec_n) <= 10:
                st.success(t("✅ Nitrogen (N) is optimal."))
            elif N_input < rec_n:
                diff = rec_n - N_input
                st.error(t(f"🔻 Nitrogen (N) is low by {diff} units. ➤ Suggest: Apply **Urea** or **DAP**."))
            else:
                diff = N_input - rec_n
                st.warning(t(f"🔺 Nitrogen (N) is high by {diff} units. ➤ Avoid further nitrogen application."))

            # Phosphorus
            if abs(P_input - rec_p) <= 10:
                st.success(t("✅ Phosphorus (P) is optimal."))
            elif P_input < rec_p:
                diff = rec_p - P_input
                st.error(t(f"🔻 Phosphorus (P) is low by {diff} units. ➤ Suggest: Apply **SSP** or **DAP**."))
            else:
                diff = P_input - rec_p
                st.warning(t(f"🔺 Phosphorus (P) is high by {diff} units. ➤ Avoid excess P fertilizers."))

            # Potassium
            if abs(K_input - rec_k) <= 10:
                st.success(t("✅ Potassium (K) is optimal."))
            elif K_input < rec_k:
                diff = rec_k - K_input
                st.error(t(f"🔻 Potassium (K) is low by {diff} units. ➤ Suggest: Apply **MOP**."))
            else:
                diff = K_input - rec_k
                st.warning(t(f"🔺 Potassium (K) is high by {diff} units. ➤ Avoid adding Potash (K)."))
        else:
            st.warning(t("⚠️ No fertilizer data found for selected crop."))
elif page == "📊 Historical Weather":
    st.header(t("📊 Historical Weather Comparison"))

    user = st.session_state['current_user']
    city = st.session_state.get("city", "")

    # --- Fetch the latest record for this user & city ---
    hist = history_col.find_one(
        {"user": user, "city": city},
        sort=[("timestamp", -1)]  # latest first
    )

    # --- Check if current session has values ---
    if all(k in st.session_state for k in ["temperature", "humidity", "rainfall"]):
        current = {
            t("Temperature (°C)"): st.session_state["temperature"],
            t("Humidity (%)"): st.session_state["humidity"],
            t("Rainfall (mm)"): st.session_state["rainfall"]
        }

        if hist:
            historical = {
                t("Temperature (°C)"): hist.get("temperature", 0),
                t("Humidity (%)"): hist.get("humidity", 0),
                t("Rainfall (mm)"): hist.get("rainfall", 0)
            }

            # --- Display comparison ---
            compare_df = pd.DataFrame([historical, current], index=[t("Previous Input"), t("Current Input")])
            st.dataframe(compare_df)

            # Plot
            fig, ax = plt.subplots()
            compare_df.T.plot(kind="bar", ax=ax)
            plt.title(t(f"Current vs Previous Weather - {city}"))
            plt.ylabel(t("Value"))
            plt.xticks(rotation=0)
            st.pyplot(fig)
        else:
            st.warning(t("⚠️ No previous data found for this user & city. Please enter data first."))
    else:
        st.warning(t("⚠️ Please enter your current weather & soil data first."))

    # --- Optional: Save current input to history for next login ---
    if all(k in st.session_state for k in ["N", "P", "K", "temperature", "humidity", "rainfall"]):
        history_col.insert_one({
            "user": user,
            "city": city,
            "N": st.session_state["N"],
            "P": st.session_state["P"],
            "K": st.session_state["K"],
            "temperature": st.session_state["temperature"],
            "humidity": st.session_state["humidity"],
            "rainfall": st.session_state["rainfall"],
            "timestamp": datetime.utcnow()
        })

elif page == "📊 Crop Prediction Insights":
    st.header(t("📊 Top Crop Prediction Insights"))
    
    if "fertilizer_recommendations" in st.session_state:
        crops = model.classes_
        input_data = [
            st.session_state['N'],
            st.session_state['P'],
            st.session_state['K'],
            st.session_state['temperature'],
            st.session_state['humidity'],
            st.session_state['ph'],
            st.session_state['rainfall']
        ]
        # --- use DataFrame with same columns as training ---
        input_df = make_input_df(input_data)
        probabilities = model.predict_proba(input_df)[0]
        top_indices = np.argsort(probabilities)[::-1][:3]

        # Pie Chart
        fig, ax = plt.subplots()
        ax.pie([probabilities[i]*100 for i in top_indices],
               labels=[crops[i] for i in top_indices],
               autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

    else:
        st.warning(t("⚠️ Please complete crop recommendation first."))

elif page == "📈 Yield & Forecast":
    st.header("📈 Yield & Profit Forecast")

    if "recommendation" not in st.session_state:
        st.warning("⚠️ Please generate a crop recommendation first.")
        st.stop()
    else:
        recommended_crop = st.session_state["recommendation"]
        rec_data = yield_df[yield_df["Crop"].str.lower() == recommended_crop.lower()]

        if not rec_data.empty:
            yield_kg = rec_data.iloc[0]["Avg_Yield_kg_per_acre"]
            default_price = rec_data.iloc[0]["Market_Price_Rs_per_kg"]
            price_per_kg = st.number_input(t("Enter Market Price (₹/kg)"), value=float(default_price))

            profit = yield_kg * price_per_kg

            st.success(t(f"📊 Expected Yield: **{yield_kg} kg/acre**"))
            st.success(t(f"💰 Estimated Profit: **₹{profit:,.2f} per acre**"))

            # Optional: Voice output
            if language != "English":
                speak(f"Expected yield is {yield_kg} kg per acre and estimated profit is ₹{profit:.2f}")

            if st.button(t("🔊 Listen to Forecast")):
                speak(t("Expected yield") + f": {yield_kg} kg/acre. " + t("Profit") + f": ₹{profit:,.2f} per acre.")
        else:
            st.warning(t("⚠️ Yield data not available for this crop."))



elif page == "♻️ Crop Rotation Plan":
    st.header(t("♻️ Crop Rotation Plan"))

    if "recommendation" not in st.session_state:
        st.warning(t("⚠️ Please generate a crop recommendation first."))
    else:
        recommended_crop = st.session_state["recommendation"]
        st.success(t(f"✅ Base Crop: **{recommended_crop}**"))

        base_type = crop_type_map.get(recommended_crop.lower())
        if base_type:
            type_order = ["legume", "cereal", "oilseed"]
            current_index = type_order.index(base_type)
            rotation_plan = []

            for i in range(1, 4):
                next_type = type_order[(current_index + i) % 3]
                candidates = rotation_cycle[next_type]
                suggested_crop = np.random.choice(candidates)
                season_label = [t("Next Season"), t("Season After"), t("3rd Season After")][i - 1]
                rotation_plan.append((season_label, next_type.capitalize(), suggested_crop))

            st.subheader(t("🔄 Multi-Season Plan"))
            for label, typ, crop in rotation_plan:
                st.write(t(f"👉 {label} ({typ}): **{crop.capitalize()}**"))
        else:
            st.warning(t("⚠️ No rotation type information available for this crop."))

        rotation_crop = rotation_rules.get(recommended_crop.lower(), None)
        if rotation_crop:
            st.info(t(f"📌 Suggested Follow-up Crop (for direct rotation): **{rotation_crop}**"))
        else:
            st.warning(t("⚠️ No crop rotation advice available."))


elif page == "📄 Download Report":
    st.header(t("📄 Download Recommendation Report"))

    if 'recommendation' in st.session_state:
        recommendation = st.session_state.recommendation
        
        # Initialize PDF
        pdf = FPDF()
        pdf.add_page()

        # Load font (DejaVuSans)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        FONT_PATH = os.path.join(BASE_DIR, "fonts", "DejaVuSans.ttf")

        pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
        pdf.add_font("DejaVu", "B", FONT_PATH, uni=True)

        pdf.set_font("DejaVu", "B", 14)
        pdf.cell(0, 10, t("Agrosense Crop Recommendation Report"), ln=True, align='C')
        pdf.ln(10)

        # Input Data
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 10, t("Input Conditions:"), ln=True)

        pdf.set_font("DejaVu", "", 12)
        pdf.cell(0, 10, t(f"Nitrogen (N): {st.session_state.N}"), ln=True)
        pdf.cell(0, 10, t(f"Phosphorus (P): {st.session_state.P}"), ln=True)
        pdf.cell(0, 10, t(f"Potassium (K): {st.session_state.K}"), ln=True)
        pdf.cell(0, 10, t(f"pH: {st.session_state.ph}"), ln=True)
        pdf.cell(0, 10, t(f"Temperature: {st.session_state.temperature} °C"), ln=True)
        pdf.cell(0, 10, t(f"Humidity: {st.session_state.humidity} %"), ln=True)
        pdf.cell(0, 10, t(f"Rainfall: {st.session_state.rainfall} mm"), ln=True)
        pdf.ln(5)

        # Recommendation
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 10, t("Crop Recommendation:"), ln=True)

        pdf.set_font("DejaVu", "", 12)
        pdf.cell(0, 10, t(f"Recommended Crop: {recommendation}"), ln=True)
        rotation_crop = rotation_rules.get(recommendation.lower(), t("Not available"))
        pdf.cell(0, 10, t(f"Suggested Crop Rotation: {rotation_crop}"), ln=True)
        pdf.ln(5)

        # Rotation Plan
        base_type = crop_type_map.get(recommendation.lower())
        if base_type:
            type_order = ["legume", "cereal", "oilseed"]
            current_index = type_order.index(base_type)

            for i in range(1, 4):
                next_type = type_order[(current_index + i) % 3]
                candidates = rotation_cycle[next_type]
                suggested_crop = np.random.choice(candidates)
                label = [t("Next Season"), t("Season After"), t("3rd Season After")][i - 1]
                pdf.cell(0, 10, t(f"{label} ({next_type.title()}): {suggested_crop}"), ln=True)
        else:
            pdf.cell(0, 10, t("No detailed rotation cycle available."), ln=True)

        pdf.ln(5)

        # Fertilizer
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(0, 10, t("Fertilizer Recommendation:"), ln=True)

        pdf.set_font("DejaVu", "", 12)
        fert = fertilizer_data.get(recommendation.lower(), t("No fertilizer info available."))
        fert = clean_pdf_text(fert)

        pdf.multi_cell(0, 10, txt=fert)
        pdf.ln(5)

        # Yield & Profit Forecast
        rec_data = yield_df[yield_df["Crop"].str.lower() == recommendation.lower()]
        if not rec_data.empty:
            yield_kg = rec_data.iloc[0]["Avg_Yield_kg_per_acre"]
            default_price = rec_data.iloc[0]["Market_Price_Rs_per_kg"]
            profit = yield_kg * default_price

            pdf.set_font("DejaVu", "B", 12)
            pdf.cell(0, 10, t("Yield & Profit Forecast:"), ln=True)

            pdf.set_font("DejaVu", "", 12)
            pdf.cell(0, 10, t(f"Expected Yield: {yield_kg} kg/acre"), ln=True)
            pdf.cell(0, 10, t(f"Estimated Profit: ₹{profit:,.2f} per acre"), ln=True)
        else:
            pdf.cell(0, 10, t("Yield data not available."), ln=True)

        # Save & Download
        pdf_bytes = bytes(pdf.output(dest="S"))
        pdf_buffer = io.BytesIO(pdf_bytes)

        st.download_button(
            label=t("Download Full Report as PDF"),
            data=pdf_buffer,
            file_name="agrosense_full_report.pdf",
            mime="application/pdf"
        )

    else:
        st.warning(t("⚠️ Please generate a recommendation first."))
