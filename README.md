# Advertising Sales Prediction

This project demonstrates how to predict the sales of a company based on their advertising budget. The dataset contains information on advertising expenses across different media channels (TV, Radio, and Newspaper) and the corresponding sales.

## Project Overview

- **Dataset Source**: Kaggle (Advertising dataset)
- **Objective**: Predict the sales based on the advertising budgets spent on TV, Radio, and Newspaper.
- **Technologies Used**:
  - Python
  - Pandas
  - Scikit-learn
  - Matplotlib / Seaborn (for visualization)
  
## Files in the Project

1. **`load_data.py`**: This script loads the dataset, performs data analysis, and prepares it for modeling.
2. **`modeling.py`**: This script builds and evaluates the machine learning model to predict sales.
3. **`visualization.py`**: This script visualizes the relationships between the features and sales.
4. **`requirements.txt`**: Contains all the necessary Python libraries for the project.
5. **`README.md`**: This file, providing an overview of the project.

## Steps Followed

### 1. **Data Loading and Preprocessing**
   - Loaded the advertising data from the CSV file.
   - Checked for missing values and duplicates.
   - Explored the basic statistical summary of the data.

### 2. **Exploratory Data Analysis (EDA)**
   - Performed exploratory data analysis (EDA) to understand the relationships between advertising budgets and sales.
   - Visualized the data using scatter plots and correlation matrices.

### 3. **Building the Model**
   - Used a Linear Regression model to predict sales based on TV, Radio, and Newspaper advertising budgets.
   - Evaluated the model using Mean Squared Error (MSE) and R-squared metrics.

### 4. **Evaluation**
   - Achieved an R-squared value of 0.91, meaning that 91% of the variance in sales is explained by the model.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/mohiittt/Advertising-Sales-Prediction.git
2. Navigate to project Directory:
    ```bash
    cd Advertising-Sales-Prediction
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
4. Run the main script:
    ```bash
    python scripts/load_data.py
## Model Performance

- **Mean Squared Error (MSE)**: 2.9077569102710896
- **R-squared**: 0.9059011844150826
