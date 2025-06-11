import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("iot_energy_data_cleaned.csv")

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=["site_type", "location_type"], drop_first=True)

# Split features and target
X = df.drop("energy_efficiency", axis=1)
y = df["energy_efficiency"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model to file
with open("rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as rf_model.pkl")
