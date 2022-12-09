import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# == DATA LOADING ============================================================================
dataset_csv = "weatherAUS"
dataset = pd.read_csv(f"{dataset_csv}.csv")

# print("shape:")
# print(dataset.shape)

X = dataset.iloc[:,:-1]     # all columns except the last, RainTomorrow
y = dataset['RainTomorrow']

# == DATA VISUALIZATION ======================================================================
# print("X:")
# print(X)
# print("\ny:")
# print(y)

# NTS: plot?

# == DATA SPLITTING ==========================================================================
# set random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=9)

print("Train shapes")
print("X_train shape:")
print(X_train.shape)
print("y_train shape:")
print(y_train.shape)

print("Test shapes")
print("X_test shape:")
print(X_test.shape)
print("y_test shape:")
print(y_test.shape)

# == DATA CLEANING ===========================================================================
# function for data cleaning
def cleanData(X_cols, y_col):
    # -- drop variables that are irrelevant and has many NA values -------------------------------
    to_drop = [
        'Evaporation',
        'Sunshine',
        'Cloud9am',
        'Cloud3pm',
        'Location',
        'Date'
    ]
    X_cols.drop(to_drop, inplace=True, axis=1)

    # -- drop records with NA values under RainToday ---------------------------------------------
    X_cols.dropna(subset=['RainToday'], inplace=True)

    print("\nAfter dropping records")
    print(X_cols.head())
    print("shape:")
    print(X_cols.shape)

    # -- label encode non-numeric variables ------------------------------------------------------
    X_cols[[
        'WindGustDir',
        'WindDir9am',
        'WindDir3pm',
        'RainToday'
        ]] = X_cols[[
        'WindGustDir',
        'WindDir9am',
        'WindDir3pm',
        'RainToday'
        ]].apply(lambda x: pd.factorize(x)[0])

    y_col = pd.factorize(y_col)[0]


    print("\nAfter label-encoding")
    print("X_cols:")
    print("\nhead:")
    print(X_cols.head())
    print("\ntail:")
    print(X_cols.tail())
    print("shape:")
    print(X_cols.shape)

    print("\ny_train:")
    print(y_col)

    # -- impute missing values -------------------------------------------------------------------
    # using mode under RainTomorrow
    y_col['RainTomorrow'].fillna(y_col['RainTomorrow'].mode(), inplace=True)

    # using mean under variables (except RainToday and RainTomorrow)
    columns = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustDir',
        'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am',
        'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
        'Pressure3pm', 'Temp9am', 'Temp3pm'
    ]
    X_cols[columns] = X_cols[columns].fillna(X_cols[columns].mean())

    print("\nAfter imputing missing values")
    print("X_cols:")
    print("\nhead:")
    print(X_cols.head())
    print("\ntail:")
    print(X_cols.tail())
    print("shape:")
    print(X_cols.shape)

    print("\ny_train:")
    print(y_col)

    # work in progress...
    # -- detect and eliminate outliers using z-score ---------------------------------------------
    # threshold = 3  # drop records greater than this value
    # X_cols[(np.abs(stats.zscore(X_cols)) < threshold).all(axis=1)]

    # print("\nAfter eliminating outliers")
    # print("\nhead:")
    # print(X_cols.head())
    # print("\ntail:")
    # print(X_cols.tail())
    # print("shape:")
    # print(X_cols.shape)

# call the function for data cleaning
print("Data cleaning for training set:")
cleanData(X_train, y_train)

