import pandas as pd

# Path to the dataset
data_path = 'C:/Users/MOHIT/OneDrive/Desktop/GroundTruth Project/data/advertising.csv'


# Load the dataset
df = pd.read_csv(data_path)

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Display dataset summary
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Check for duplicate values
print("\nDuplicate Values in Each Column:")
print(df.duplicated().sum())

# Check for outliers
print("\nOutliers in Each Column:")
print(df.describe().T)

import matplotlib.pyplot as plt
import seaborn as sns

# Plot a pairplot to visualize relationships
sns.pairplot(df)
plt.show()

# Visualize the correlation between variables
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define features (X) and target variable (y)
X = df[['TV', 'Radio', 'Newspaper']]  # Features
y = df['Sales']  # Target variable (Sales)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict the sales on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared (R2 score)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()

import joblib

# Save the model to a file
joblib.dump(model, 'advertising_sales_model.pkl')
