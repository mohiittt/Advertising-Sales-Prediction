import pandas as pd

# Load dataset
data_path = 'C:/Users/MOHIT/OneDrive/Desktop/Advertising-Sales-Prediction/data/advertising.csv'
df = pd.read_csv(data_path)

# Show first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Data Info
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Check for duplicate values
print("\nDuplicate Values in Each Column:")
print(df.duplicated().sum())

# Basic Statistical Analysis
print("\nOutliers in Each Column:")
print(df.describe())

