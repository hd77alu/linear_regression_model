# linear Regression Model to Predict CO2 emissions in East Africa

## Mission And Problem
- My mission focuses on Climate Change and how to use technologies to address environmental challenges in Africa.
- The problem addressed here is the prediction of CO2 emissions trends to support improved climate-mitigation planning.
- The goal is to contribute practical, data-driven tools that support Africa's efforts in effective climate adaptation.

## Dataset Information
This project uses historical country-level CO2 emissions and socioeconomic indicators from 2000 to 2020 to create a model that predicts total CO2 emissions excluding LUCF in East Africa.


**Dataset:** `africa-co2-emissions.csv`

### Dataset Characteristics:
- **Rows:** 1,134
- **Columns:** 20 total (3 non-numeric, 17 numeric)
- **Source:** [African countries CO2 emissions data](https://www.kaggle.com/datasets/victoraiyewumi/co2-emission-africa)


## Project Structure
```text
├── summative/
	├── API/
		prediction.py
	├── FlutterApp/
	├── linear_regression/
		multivariate.ipynb
		├── data/
			africa-co2-emissions.csv
		├── final_model/
			best_linear_regression_model.joblib
```

## Setup Instructions

1. Clone or Download the Repository
```bash
git clone https://github.com/hd77alu/linear_regression_model
cd linear_regression_model
```
2. Create and activate a Python virtual environment.
3. Install dependencies:
	 - `pip install numpy pandas scikit-learn matplotlib joblib`

## How to Use the Notebook

### Method 1: Using Jupyter Notebook (Local)
```bash
# Navigate to project directory
cd linear_regression_model

# Launch Jupyter Notebook
jupyter notebook

# Open the file: multivariate.ipynb
```

### Method 2: Using Google Colab
1. Click the "Open in Colab" badge at the top of the notebook
2. Upload `africa-co2-emissions.csv` to your files
3. Run all cells

### Method 3: Using VS Code
1. Open VS Code
2. Install the Jupyter extension (if not already installed)
3. Open the folder `linear_regression`
4. Click on `multivariate.ipynb`
5. Select Run all
6. Select Virtual Python kernel when prompted

## Project Implementation

### 1. Problem Framing And Scope
This implementation is built around a climate-focused regression task: estimate East Africa Total CO2 Emission excluding LUCF (Mt) from historical economic and sectoral emission indicators.
The geographic scope was intentionally constrained to Eastern Africa to keep the model aligned with the mission objective and to avoid mixing very different regional emission dynamics.

### 2. Data Processing Pipeline
The notebook implementation follows a structured preprocessing pipeline:
1. Load the dataset from `summative/linear_regression/data/africa-co2-emissions.csv`.
2. Replace placeholder missing values (`N/A`, `na`, empty strings) with nulls.
3. Filter records to Eastern Africa only.
4. Convert selected columns to numeric types using safe coercion.
5. Drop highly sparse columns (such as Fugitive Emissions) when needed.
6. Fill the remaining numeric missing values with median imputation.

This step ensures that the model receives clean, consistent numeric inputs and that downstream analysis is reproducible.

### 3. Exploratory Data Analysis (EDA)
EDA was used to validate assumptions and guide feature decisions:
1. Correlation heatmap to inspect relationships between predictors and the target.

![Correlation heatmap](https://github.com/hd77alu/Portfolio101/blob/a64e5b063f134628598806aa80e6a1d73e127f36/images/eastAfrica-dataset-heatmap.png)

2. Histograms to understand the distribution and spread of key variables.

![Histograms for variable distributions](https://github.com/hd77alu/Portfolio101/blob/5724876240ffa2bb261a861e1d484fb160a3991e/images/eastAfrica-dataset-variable-distributions.png)

3. Scatter plots to inspect directional patterns and potential linear relationships.

![Scatterplots (relationship view)](https://github.com/hd77alu/Portfolio101/blob/5724876240ffa2bb261a861e1d484fb160a3991e/images/eastAfrica-dataset-scatterplots.png)

**These visualizations were not only descriptive; they directly informed which variables were likely to be useful and where multicollinearity/leakage risks might appear.**

### 4. Feature Engineering And Selection
Feature engineering was implemented with explicit decision logic:
1. Build an intermediate feature-engineering dataframe.
2. Drop columns with redundancy or leakage risk.
3. Create derived features (for example, population density) and test their usefulness.
4. Rank numeric candidates by correlation strength to the target.
5. Finalize a decomposition-aware set of predictors for training.

Final modeling features:
- Country
- Year
- Population
- Transportation (Mt)
- Manufacturing/Construction (Mt)
- Electricity/Heat (Mt)
- Building (Mt)

This made the model easier to interpret while reducing unstable feature overlap.

### 5. Data Standardization And Encoding
To prepare features for model training:
1. Separate features (`X`) and target (`y`).
2. Standardize numeric predictors with `StandardScaler`.
3. One-hot encode the categorical `Country` feature.
4. Split into training and testing sets using an 80/20 ratio.

This ensures numeric features are on comparable scales and categorical information is represented in a model-friendly way.

### 6. Model Training Strategy
Three regression models were trained on the same processed dataset:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

All three were trained and evaluated under the same split and preprocessing path for fair comparison.

### 7. Evaluation, Selection, And Validation
Model comparison used Mean Squared Error (MSE) as the primary metric.
The implementation includes:
1. MSE calculation for each model.
2. Ranking models by lowest test MSE.
3. Additional quick prediction checks (actual vs predicted samples).
4. Train-vs-test loss visualization for comparison across models.

The best-performing model (lowest loss) was selected automatically from the evaluation table.

### 8. Visualization of the Final Linear Fit

![Linear Regression Fitted Line on Test Data](https://github.com/hd77alu/Portfolio101/blob/5724876240ffa2bb261a861e1d484fb160a3991e/images/eastAfrica-dataset-fitted-line.png)

A dedicated scatter plot was implemented to show the fitted linear relationship after training the Linear Regression model.
For clarity, the fitted line is visualized against a chosen feature slice (Transportation (Mt)) while other features are held at baseline values.
This provides a readable 2D interpretation of a multivariate model.

### 9. Model Persistence And Inference Script
After selection, the best model is saved to:
- `summative/linear_regression/final_model/best_linear_regression_model.joblib`

To support future backend/API integration, a standalone inference script was built:
- `summative/API/prediction.py`

The script:
1. Loads the saved model.
2. Rebuilds the same preprocessing logic used during training.
3. Validates typed input payloads.
4. Supports both single and batch predictions.
5. Returns prediction output in JSON-friendly format.
