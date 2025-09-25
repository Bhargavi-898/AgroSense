Agrosense – Smart Crop Recommendation System

Agrosense is a web-based agricultural assistant designed to help farmers and agriculturists make informed decisions about crop selection, soil health, fertilizer usage, and crop rotation. It combines soil data, weather information, seasonal trends, and machine learning models to provide actionable insights, improving productivity and sustainability.

🌟 Features
1. Home Dashboard

Welcomes the user and provides a quick overview of Agrosense functionalities.

Lists key benefits such as:

✅ Soil Nutrient Analysis (N, P, K, pH)

✅ Real-Time Weather Updates (Temperature, Humidity, Rainfall)

✅ Crop Recommendation

✅ Fertilizer Suggestions

✅ Crop Rotation Planning

✅ PDF Reports

2. Input Soil & Weather Data

Users can input:

Soil nutrients: Nitrogen (N), Phosphorus (P), Potassium (K), pH

Weather conditions: Temperature, Humidity, Rainfall

Users can either:

Fetch live weather data for a specific city.

Input manual weather data using sliders.

Data is stored in the session state for further analysis.

3. Crop Recommendation

Machine learning model predicts best-suited crops based on soil and weather input.

Provides:

Top crop recommendation with probability

Top 3 crop options for better choice

Displays current agricultural season (Kharif, Rabi, Zaid).

4. Fertilizer Suggestion

Based on the recommended crop and soil nutrients:

Suggests optimal NPK fertilizer levels.

Alerts users if nutrients are low, optimal, or high.

Recommends specific fertilizers (e.g., Urea, DAP, MOP, SSP).

Helps prevent over-fertilization or nutrient deficiency.

5. Historical Weather Comparison

Compares current weather conditions with historical averages for the same month.

Provides visual insights using bar charts.

Helps farmers understand weather trends and plan crop schedules.

6. Crop Prediction Insights

Shows top crop predictions with probability distribution.

Visualized as a pie chart for clarity.

Helps users understand alternative crop options.

7. Crop Rotation Plan

Provides a multi-season crop rotation plan to maintain soil health.

Based on crop type (legume, cereal, oilseed) and previous crop.

Suggests next season crops, season after, and third season after.

Includes optional direct rotation suggestions.

8. Download Recommendation Report

Users can download a comprehensive PDF report including:

Input soil and weather conditions

Recommended crop and alternatives

Crop rotation plan

Fertilizer recommendations

Yield and profit forecasts

Report is ready for printing or sharing.

🛠️ Technologies Used

Frontend: Streamlit
 – Interactive UI for data input and visualization.

Backend: Python – Handles logic, predictions, and PDF generation.

Machine Learning: Scikit-learn (RandomForestClassifier) – Crop recommendation model.

Database/Files:

historical_df – Historical weather data

fertilizer_data – Fertilizer information per crop

yield_df – Average yield and market price data

rotation_rules & rotation_cycle – Crop rotation rules

Libraries:

numpy – Numerical calculations

pandas – Data manipulation

matplotlib – Charts for weather and crop insights

FPDF – PDF report generation

Language Translation:

t() function integrated to translate UI and reports for multilingual support.

⚙️ Workflow

User Inputs Data:

Soil nutrients and weather conditions.

Fetches live weather if needed.

Model Predicts Crop:

ML model suggests best crop(s).

Stores top recommendations in session.

Fertilizer Guidance:

Compares soil nutrients with recommended NPK levels.

Suggests fertilizers and alerts for deficiency/excess.

Weather & Insights:

Displays historical vs current weather.

Provides probability distribution for crop choices.

Crop Rotation:

Suggests multi-season crop rotation.

Helps improve soil fertility and reduce disease risk.

PDF Report:

All recommendations and insights saved in a downloadable report.

📈 Benefits

Helps small and large farmers make data-driven decisions.

Prevents waste of fertilizers and crop loss due to unsuitable planting.

Supports sustainable agriculture with crop rotation planning.

Provides insights and reports in an easy-to-understand format.

🌍 Future Enhancements

Integration with real-time satellite soil data.

Include disease prediction for crops.

Mobile-friendly version with SMS or WhatsApp alerts.

Integration of AI-powered weather forecasts.