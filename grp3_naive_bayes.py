import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

# == DATA LOADING ================================================================================
dataset_csv = "weatherAUS"
dataset = pd.read_csv(f"{dataset_csv}.csv")

# print("shape:")
# print(dataset.shape)

X = dataset.iloc[:,:-1]     # all columns except the last, RainTomorrow
y = dataset['RainTomorrow'].to_frame()  # RainTomorrow

# == DATA VISUALIZATION ==========================================================================
# print("X:")
# print(X)
# print("\ny:")
# print(y)

# NTS: plot?

# == DATA SPLITTING ==============================================================================
# set random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=41)

print("---- Train shapes")
print("X_train shape:")
print(X_train.shape)
print("y_train shape:")
print(y_train.shape)

print("Test shapes")
print("X_test shape:")
print(X_test.shape)
print("y_test shape:")
print(y_test.shape)

# == DATA CLEANING ===============================================================================
# function for data cleaning
def cleanData(set):
    # -- drop variables that are irrelevant and has many NA values -------------------------------
    to_drop = [
        'Evaporation',
        'Sunshine',
        'Cloud9am',
        'Cloud3pm',
        'Location',
        'Date'
    ]
    set.drop(to_drop, inplace=True, axis=1)

    # -- drop records with NA values under RainToday ---------------------------------------------
    set.dropna(subset=['RainToday'], inplace=True)

    print("\n---- After dropping records")
    print(set.head())
    print("shape:")
    print(set.shape)

    # -- label encode non-numeric variables ------------------------------------------------------
    columns = [
        'WindGustDir',
        'WindDir9am',
        'WindDir3pm',
        'RainToday',
        'RainTomorrow'
    ]
    set[columns] = set[columns].apply(lambda x: pd.factorize(x)[0])

    print("\n---- After label-encoding")
    print("\nhead:")
    print(set.head())
    # print("\ntail:")
    # print(set.tail())
    print("shape:")
    print(set.shape)

    # -- impute missing values -------------------------------------------------------------------
    # using mode under RainTomorrow
    set['RainTomorrow'].fillna(set['RainTomorrow'].mode(), inplace=True)

    # using mean under variables (except RainToday and RainTomorrow)
    columns = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustDir',
        'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am',
        'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
        'Pressure3pm', 'Temp9am', 'Temp3pm'
    ]
    set[columns] = set[columns].fillna(set[columns].mean())

    print("\n---- After imputing missing values")
    print("\nhead:")
    print(set.head())
    # print("\ntail:")
    # print(set.tail())
    print("shape:")
    print(set.shape)

    # -- detect and eliminate outliers using z-score ---------------------------------------------
    threshold = 3  # drop records greater than this value
    set = set[(np.abs(stats.zscore(set)) < threshold).all(axis=1)]

    print("\n---- After eliminating outliers")
    print("\nhead:")
    print(set.head())
    # print("\ntail:")
    # print(set.tail())
    print("shape:")
    print(set.shape)

    return set

# combine X_train and y_train for data cleaning
train_set = pd.concat([X_train,y_train], axis=1)

print("\n---- Train set after concatenating X_train and y_train")
print(train_set)

# call the function for data cleaning
train_set = cleanData(train_set)
print("\n---- After data cleaning for training set:")
print(train_set)

