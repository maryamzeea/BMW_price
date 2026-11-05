# BMW Used Car Price Prediction

This project implements an end-to-end Machine Learning pipeline in Python to analyze and predict used BMW car prices. The workflow encompasses Exploratory Data Analysis (EDA), One-Hot Encoding for categorical variables, Feature Scaling, statistical hypothesis testing (Pearson correlation and Chi-Square tests for feature selection), and Multiple Linear Regression evaluation.

## Project Overview

* **Goal**: Build a baseline predictive regression model to estimate the price of used BMW vehicles based on attributes like vehicle age, mileage, engine size, fuel efficiency ($mpg$), transmission, and model type.

* **Model**: Multiple Linear Regression (`sklearn.linear_model.LinearRegression`).

* **Evaluation Metrics**: $R^2$ Score and Adjusted $R^2$ Score.

## Project Architecture & Workflow

1. **Exploratory Data Analysis (EDA)**

   * Initial inspection (`head`, `describe`, `isnull`).

   * Visualizing relationships using `seaborn` heatmaps, boxplots (Year vs Price, Engine Size vs Price), scatter plots (Mileage vs Price), and histograms.

2. **Data Preprocessing & Encoding**

   * Identification of categorical features (`model`, `transmission`, `fuelType`).

   * One-Hot Encoding (`pd.get_dummies`) with `drop_first=True` to eliminate multi-collinearity issues.

3. **Feature Scaling**

   * Applied `StandardScaler` on numerical features: `year`, `mileage`, `engineSize`, `mpg`, and `tax`.

4. **Statistical Feature Selection**

   * **Pearson Correlation**: Evaluated linear relationships between continuous numerical variables and vehicle `price`.

   * **Chi-Square (**$\chi^2$**) Test of Independence**: Tested dummy categorical variables against `price` ($\alpha = 0.05$) to filter out statistically insignificant features.

5. **Model Training & Evaluation**

   * **Data Split**: 80% Training set, 20% Testing set (`random_state=42`).

   * **Model**: Ordinary Least Squares (OLS) Linear Regression.

   * **Performance**: Calculated $R^2$ and Adjusted $R^2$ to account for model complexity ($P$ features, $n$ samples).

## Prerequisites & Installation

Ensure you have Python 3.x installed along with the required libraries:

```
pip install pandas numpy seaborn matplotlib scikit-learn scipy

```

## How to Run

1. Place the dataset `bmw.csv` in your working directory.

2. Execute the script or Jupyter Notebook containing the workflow code:

```
python main.py

```

## Project Structure

```
.
├── bmw.csv               # Dataset containing BMW used car listings
├── main.py               # Main pipeline script (preprocessing, EDA, modeling)
└── README.md             # Project documentation

```

##  Results & Performance Formulae

* **Coefficient of Determination (**$R^2$**):** Evaluates overall goodness of fit on test predictions $y_{pred}$ against ground truth $y_{test}$.

* **Adjusted** $R^2$ **Formula:**

$$
Adjusted\ R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - P - 1}
$$

Where:

* $n$ = Number of training samples ($X_{train}.shape[0]$)

* $P$ = Number of predictors/features ($X_{train}.shape[1]$)