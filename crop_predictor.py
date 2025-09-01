import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")  # You’ll add this CSV file next

# Prepare data
X = df.drop('label', axis=1)
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Predict Top 3 Crops
def get_top_3_crops(input_data):
    probabilities = model.predict_proba([input_data])[0]
    top_indices = np.argsort(probabilities)[::-1][:3]
    crops = model.classes_
    return [(crops[i], round(probabilities[i]*100, 2)) for i in top_indices]

# Example input: [N, P, K, temperature, humidity, ph, rainfall]
sample_input = [90, 42, 43, 20.5, 80, 6.5, 200]
top_crops = get_top_3_crops(sample_input)

print("\nTop 3 Recommended Crops:")
for crop, score in top_crops:
    print(f"{crop}: {score}% confidence")
