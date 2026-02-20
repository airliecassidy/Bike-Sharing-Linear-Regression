# Bike-Sharing-Linear-Regression

This project is implemented in a **Jupyter Notebook** and fits a linear regression model to the Bike Sharing Dataset from the UCI Machine Learning Repository.

Bike sharing systems generate rich, timestamped usage data that makes them well-suited for predictive modelling. This project fits a multiple linear regression model to the daily aggregated dataset, treating total rental count (cnt) as the target variable and using a combination of weather, temporal, and categorical features as predictors.

The goal is to understand which factors most strongly drive daily ridership and to build a model which generates well on unseen data. 

## Dataset
Download the **Day** dataset from:  
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
 

## Setup
Before running the notebook, update the following variable to point to the location of the downloaded dataset on your machine:

```python
data_path = "path/to/day.csv"

```

# Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

# Methodology
The notebook wals through the entire modelling pipeline: data analysis, preprocessing, model fitting, evaluation. 

The goal was to explore how deliberate feature design can improve the predictive performance of a linear model, demonstrating the key distinction between linearity in model parameters and linearity in input features.
Linear regression is implemented from scratch using NumPy (closed-form least squares), without relying on scikit-learn's regression classes. Two models are trained and compared:

Baseline model — trained on 15 cleaned input features
Engineered model — trained on an expanded 33-feature design matrix with nonlinear and interaction terms

The model is fit using the closed-form normal equation implemented in NumPy. The feature matrix is augmented with a bias column of ones. An 80/20 train-test split is used with a fixed random seed for reproducibility. Model performance is evaluated using Mean Squared Error (MSE) on both sets.

# Limitations
Despite the MSE improvements, several limitations remain. The model still produces large absolute MSE values, reflecting the inherent complexity of the dataset. Feature engineering introduced multicollinearity — particularly between atemp and its derived variants — which can destabilize individual coefficient estimates even while improving predictive performance overall. Heteroscedasticity is also visible in the residual plots, with larger prediction errors at low ridership levels. Finally, results are based on a single train-test split, which introduces variance in the evaluation.

# Potential Extensions
Future improvements could include spline-based temperature features to better capture saturation effects, Ridge or Lasso regularization to manage multicollinearity from feature engineering, season-specific sub-models, and repeated cross-validation for more robust performance estimation.


