"""
Task 4 - Sales Prediction using Advertising Data
----------------------------------------------------
Predicts product Sales based on advertising spend across
TV, Radio, and Newspaper channels using Linear Regression
(and compares with other regression models).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style("whitegrid")

# ---------------------------------------------------
# 1. Load Data
# ---------------------------------------------------
DATA_PATH = "Advertising.csv"   # change path if needed
df = pd.read_csv(r"C:\Users\ADMIN\Downloads\archive (5)\Advertising.csv")

# Drop the unnamed index column if present
if df.columns[0].strip() == "" or "Unnamed" in df.columns[0]:
    df = df.drop(columns=[df.columns[0]])

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nSummary statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())

# ---------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------
print("\nCorrelation with Sales:\n", df.corr()["Sales"].sort_values(ascending=False))

# Pairplot: relationship between each ad channel and Sales
sns.pairplot(df, x_vars=["TV", "Radio", "Newspaper"], y_vars="Sales",
             height=4, aspect=1, kind="reg")
plt.suptitle("Advertising Spend vs Sales", y=1.02)
plt.savefig("ad_spend_vs_sales.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# ---------------------------------------------------
# 3. Prepare Features & Target
# ---------------------------------------------------
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# 4. Train & Compare Models
# ---------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = r2
    print(f"{name:18s} | RMSE: {rmse:6.3f} | MAE: {mae:6.3f} | R2 Score: {r2:6.3f}")

# ---------------------------------------------------
# 5. Best Model Details
# ---------------------------------------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_preds = best_model.predict(X_test)

print("\n" + "=" * 55)
print(f"BEST MODEL: {best_model_name}")
print("=" * 55)

if best_model_name == "Linear Regression":
    print("Intercept:", best_model.intercept_)
    print("Coefficients:")
    for feature, coef in zip(X.columns, best_model.coef_):
        print(f"  {feature:10s}: {coef:.4f}")
    print("\nInterpretation: For every extra unit (e.g. $1000) spent on a channel,")
    print("Sales is predicted to change by its coefficient amount, holding other channels constant.")

# Actual vs Predicted plot
plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_preds, alpha=0.7, edgecolor="k")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(f"Actual vs Predicted Sales ({best_model_name})")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

print("\nSaved charts: ad_spend_vs_sales.png, correlation_heatmap.png, actual_vs_predicted.png")

# ---------------------------------------------------
# 6. Predict Sales for New Advertising Budget
# ---------------------------------------------------
new_budget = pd.DataFrame({"TV": [150], "Radio": [25], "Newspaper": [10]})
predicted_sales = best_model.predict(new_budget)[0]
print(f"\nPredicted Sales for TV=150, Radio=25, Newspaper=10 budget: {predicted_sales:.2f} units")
