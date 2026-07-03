import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
df = pd.read_csv("dataset/HDI.csv")

# Keep only numeric columns
df = df.select_dtypes(include=["number"])

# Remove columns with all missing values
df = df.dropna(axis=1, how="all")

# Target column
target = "Human Development Index (2021)"

# Feature columns
features = [
    "Life Expectancy at Birth (2021)",
    "Expected Years of Schooling (2021)",
    "Gross National Income Per Capita (2021)"
]

# Remove rows where target is missing
df = df.dropna(subset=[target])

# Select features and target
X = df[features]
y = df[target]

# Fill missing values
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/hdi_model.pkl")
joblib.dump(imputer, "model/imputer.pkl")

print("Model trained successfully!")