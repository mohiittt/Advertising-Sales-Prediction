import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_path = 'C:/Users/MOHIT/OneDrive/Desktop/Advertising-Sales-Prediction/data/advertising.csv'
df = pd.read_csv(data_path)

# Plot pairplot to see relationship between features and target
sns.pairplot(df)
plt.show()

# Plot correlation heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
