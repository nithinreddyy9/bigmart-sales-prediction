import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Train.csv")

# Show dataset info
print(df.head())

# Remove missing values
df = df.dropna()

# Features and target
X = df.drop("Item_Outlet_Sales", axis=1)
y = df["Item_Outlet_Sales"]

# Convert categorical columns into numeric
X = pd.get_dummies(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor()

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)

print("RMSE:", rmse)
print("R2 Score:", r2)
