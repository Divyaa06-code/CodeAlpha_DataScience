# CodeAlpha_DataScience

# Iris Flower Classification 🌸

This project uses the famous **Iris dataset** to train a machine
learning model that can classify Iris flowers into three species: -
Iris-setosa - Iris-versicolor - Iris-virginica

## 📊 Dataset

The dataset contains 150 samples with the following features: - Sepal
Length (cm) - Sepal Width (cm) - Petal Length (cm) - Petal Width (cm)

Target variable: - Species (Setosa, Versicolor, Virginica)

## 🚀 Steps in the Project

1.  Load the Iris dataset using **Scikit-learn** or CSV file.
2.  Perform **data exploration and visualization** using pandas,
    matplotlib, seaborn.
3.  Split the dataset into **training and testing sets**.
4.  Train classification models (e.g., Logistic Regression, Decision
    Tree, Random Forest, SVM).
5.  Evaluate the model using metrics such as **accuracy, confusion
    matrix, classification report**.
6.  Make predictions on new data.

## ⚙️ Technologies Used

-   Python 🐍
-   Pandas
-   NumPy
-   Matplotlib & Seaborn
-   Scikit-learn

## 📈 Example Output

-   Model Accuracy: \~95% (varies by algorithm)
-   Confusion matrix and classification report
-   Example prediction: *Input -\> \[5.1, 3.5, 1.4, 0.2\], Output -\>
    Iris-setosa*

## ▶️ How to Run

1.  Clone the repository or copy the notebook.

2.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

3.  Open Jupyter Notebook:

    ``` bash
    jupyter notebook
    ```

4.  Run all cells to train and test the model.

## 📌 Future Improvements

-   Try different algorithms (KNN, Gradient Boosting, Neural Networks).
-   Deploy the model as a **web app** using Flask or Streamlit.
-   Hyperparameter tuning for better performance.


# 🚗 Car Price Prediction using Machine Learning

## 📌 Project Overview

This project predicts the **selling price of used cars** based on their
features such as year, present price, kilometers driven, fuel type,
transmission, and number of previous owners.\
We use **Linear Regression** as the machine learning model.

The dataset used is `car data.csv`.

------------------------------------------------------------------------

## 🗂 Dataset Description

The dataset contains the following columns:

-   `Car_Name` → Name of the car (dropped in preprocessing)\
-   `Year` → Year of manufacture\
-   `Selling_Price` → Price the owner wants to sell the car (Target
    variable)\
-   `Present_Price` → Current ex-showroom price (in lakhs)\
-   `Kms_Driven` → Distance driven (in kilometers)\
-   `Fuel_Type` → Petrol / Diesel / CNG\
-   `Seller_Type` → Dealer / Individual\
-   `Transmission` → Manual / Automatic\
-   `Owner` → Number of previous owners

------------------------------------------------------------------------

## ⚙️ Steps Followed

1.  **Import Libraries**

    ``` python
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    ```

2.  **Load Dataset**

    ``` python
    df = pd.read_csv("car data.csv")
    ```

3.  **Data Preprocessing**

    -   Dropped `Car_Name` (not useful for prediction)\
    -   Created new feature `Car_Age = 2025 - Year`\
    -   Dropped `Year` column\
    -   Converted categorical variables into dummy/indicator variables

    ``` python
    df = df.drop(['Car_Name'], axis=1)
    df['Car_Age'] = 2025 - df['Year']
    df = df.drop(['Year'], axis=1)
    df = pd.get_dummies(df, drop_first=True)
    ```

4.  **Split Dataset**

    ``` python
    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

5.  **Train Model**

    ``` python
    model = LinearRegression()
    model.fit(X_train, y_train)
    ```

6.  **Evaluate Model**

    ``` python
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    ```

------------------------------------------------------------------------

## 📊 Results

-   The model outputs the **predicted car prices** based on input
    features.\
-   Evaluation metrics include **Mean Squared Error (MSE)** and **R²
    Score**.\
-   Example output:

```{=html}
<!-- -->
```
    Mean Squared Error: 2.31  
    R² Score: 0.91  

------------------------------------------------------------------------

## 🚀 How to Run

1.  Clone this project or copy the code into a Jupyter Notebook.\
2.  Place `car data.csv` in the same folder as your notebook.\
3.  Run all cells step by step.\
4.  Modify input values to test your own predictions.

Example Prediction:

``` python
example_input = [X_test.iloc[0]]
print("Predicted Selling Price:", model.predict(example_input)[0])
```

------------------------------------------------------------------------

## 🛠️ Requirements

-   Python 3.8+\
-   pandas\
-   scikit-learn\
-   Jupyter Notebook

Install dependencies:

``` bash
pip install pandas scikit-learn notebook
```

------------------------------------------------------------------------

## 📌 Future Improvements

-   Try **Random Forest Regression** for better accuracy\
-   Add feature scaling for better performance\
-   Deploy the model using **Flask / Streamlit**
