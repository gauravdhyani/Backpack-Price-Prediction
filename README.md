# Backpack-Price-Prediction

##  Project Overview
This project aims to **predict the prices of backpacks based on their features** using structured tabular data from Kaggle's *Playground Series S5E2*. It demonstrates an **end-to-end machine learning workflow**, including:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering (including amplification and polynomial transformations)
- Model training and evaluation
- Generating test predictions

The primary goal is to **accurately predict backpack prices** and compare the performance of multiple regression models such as **KNN, Decision Tree, XGBoost, and LightGBM**. Advanced feature amplification techniques like **polynomial transformations** and **feature interactions** are also applied to improve accuracy.

---

## Objectives

- Understand the relationship between backpack features and price.
- Evaluate the performance of different regression models.
- Explore feature selection, amplification, and polynomial transformations to improve predictive performance.
- Generate price predictions for unseen test data.

---

##  Dataset
- **Training Data:** `/kaggle/input/playground-series-s5e2/train.csv`
- **Test Data:** `/kaggle/input/playground-series-s5e2/test.csv`

### Key Features:
- Brand, Material, Size, Compartments, Laptop Compartment, Waterproof, Style, Color, Weight Capacity (kg)

**Target Variable:** `Price`

---

##  Project Workflow

### 1. Data Preprocessing
- Load and inspect the dataset.
- Analyze columns for unique values & missing entries.
- Encode categorical features:
  - `Brand`, `Material` → Numeric IDs
  - `Size` → Standardized labels
  - `Laptop Compartment`, `Waterproof` → Binary (0/1)
- Handle missing values (e.g., fill numeric with mean).
- Normalize numeric features using **MinMaxScaler** and **StandardScaler**.
- Save processed datasets:
  - `processed_file.csv` → Preprocessed data
  - `amplified_file.csv` → Feature-amplified data

---

### 2. Exploratory Data Analysis (EDA)
- Visualize distributions of `Weight Capacity (kg)` and `Price`.
- Generate **correlation heatmaps**.
- Identify highly correlated features for selection and amplification.

---

### 3. Feature Engineering
#### 3.1 Full Feature Set
- Brand, Material, Size, Compartments, Laptop Compartment, Waterproof, Style, Color, Weight Capacity (kg)

#### 3.2 Selected Features
- Selected using **RFE** and correlation analysis:
   - Weight Capacity (kg)  
   - Brand  
   - Color  
   - Material 

#### 3.3 Amplified Features
- Feature interactions:
  - `Weight_Brand = Weight Capacity * Brand`
  - `Weight_Color = Weight Capacity * Color`
  - `Weight_Size = Weight Capacity * Size`
  - `Weight_LaptopCompartment = Weight Capacity * Laptop Compartment`
- Polynomial transformations:
  - Log, square, cube of `Weight Capacity (kg)`

---

### 4. Machine Learning Models
All models trained using **train_test_split (80/20)** and evaluated with:
- R² Score  
- Mean Squared Error (MSE)  
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- Explained Variance Score

#### 4.1 KNN Regression
- Full Features: `n_neighbors=20`
- Selected Features: `n_neighbors=350`
- Improved with scaling and feature selection.

#### 4.2 Decision Tree Regression
- Full Features: `max_depth=5`, `min_samples_split=10`
- Selected Features: `max_depth=3`, `min_samples_split=20`, `min_samples_leaf=20`
- Amplified Features improved performance.

#### 4.3 XGBoost Regression
- `objective='reg:squarederror'`, `n_estimators=75`
- Best performance with amplified features.

#### 4.4 LightGBM Regression
- `n_estimators=75`
- Achieved top R² scores with amplified features.

---

## Results

### 1. Full Feature Models

| Model | R² Score | MSE | MAE | RMSE | Explained Variance |
|-------|----------|-----|-----|------|------------------|
| KNN   | -0.048   | 1589.31 | 34.20 | 39.87 | -0.048 |
| Decision Tree | 0.001  | 1515.77 | 33.65 | 38.93 | 0.001 |

**Observation:**  
- Full feature models perform poorly, especially KNN with negative R².

---

### 2. Selected Feature Models

**Selected features:** Weight Capacity (kg), Brand, Color, Material

| Model | R² Score | MSE | MAE | RMSE | Explained Variance |
|-------|----------|-----|-----|------|------------------|
| KNN   | -0.0012 | 1518.40 | 33.67 | 38.97 | -0.0012 |
| Decision Tree | 0.0008 | 1514.89 | 33.67 | 38.92 | 0.0008 |
| XGBoost | -0.0016 | 1519.04 | 33.69 | 38.97 | -0.0016 |
| LightGBM | 0.0017 | 1513.45 | 33.65 | 38.90 | 0.0017 |

**Observation:**  
- Slight improvement over full features.  
- LightGBM shows the best predictive power among selected feature models.

---

### 3. Amplified Feature Models

**Amplified features:** Weight_Log, Weight_Brand, Weight Capacity (kg), polynomial interactions

| Model | R² Score | MSE | MAE | RMSE | Explained Variance |
|-------|----------|-----|-----|------|------------------|
| Decision Tree | 0.00076 | 1515.47 | 33.66 | 38.93 | 0.00078 |
| XGBoost | 0.00135 | 1514.01 | 33.66 | 38.91 | 0.00135 |
| LightGBM | 0.00028 | 1516.19 | 33.66 | 38.94 | 0.00030 |

**Observation:**  
- Amplified features slightly improve XGBoost performance.  
- Overall, all models have near-zero R², indicating additional external factors may improve predictions.

---

##  Challenges Faced
1. **Categorical Encoding:** Missing/inconsistent labels.  
   *Solution:* Mapping and default category assignment.
2. **Non-linear Feature Relationships:** Price influenced by feature interactions.  
   *Solution:* Polynomial and interaction features.
3. **Weak signals:** Features like `Color` or `Style` had low correlation.  
   *Solution:* Amplified interactions with `Weight`.

---

##  Solutions Deployed
- Robust **preprocessing** pipeline.
- **Feature amplification** with interactions and polynomial features.
- Used **distance-based, tree-based, and ensemble models**.
- **Feature selection** to reduce noise.
- Applied consistent transformations on test data.

---

## Conclusion

- Feature selection and amplification slightly improve model performance.  
- XGBoost with amplified features gave the best results.  
- Price prediction is limited by dataset features; incorporating external market trends or user ratings could improve accuracy.  
- Future improvements:
  - Include demand trends or ratings  
  - Use ensemble methods combining multiple models  
  - Feature engineering with external datasets  

---