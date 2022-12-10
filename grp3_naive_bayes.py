import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# == DATA LOADING ================================================================================
# in local machine
dataset_csv = "weatherAUS"
dataset = pd.read_csv(f"{dataset_csv}.csv")

# in colab from github
# csv_url = 'https://github.com/stareginio/group-3-project-cs-elec-1a/blob/main/weatherAUS.csv?raw=true'
# dataset = pd.read_csv(csv_url)

# print("shape:")
# print(dataset.shape)

X = dataset.iloc[:,:-1]     # all columns except the last, RainTomorrow
y = dataset['RainTomorrow'].to_frame()  # RainTomorrow

# print("X:")
# print(X)
# print("\ny:")
# print(y)

# == DATA SPLITTING ==============================================================================
# set random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=41)

# print("---- Train shapes")
# print("X_train shape:")
# print(X_train.shape)
# print("y_train shape:")
# print(y_train.shape)

# print("Test shapes")
# print("X_test shape:")
# print(X_test.shape)
# print("y_test shape:")
# print(y_test.shape)

# == DATA CLEANING ===============================================================================
# combine X_train and y_train for data cleaning
train_set = pd.concat([X_train,y_train], axis=1)
# print("\n---- Train set after concatenating X_train and y_train")
# print(train_set)

# parameters to be used for test set later
mode_rt = None
mean = None
mode_wgd = None
mode_wdn = None
mode_wdt = None

# function for data cleaning
def cleanData(set, mode_rt, mean, mode_wgd, mode_wdn, mode_wdt, type_set):
    print(f"\ntype_set: {type_set} ===============")
    
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

    # print("\n---- After dropping records")
    # print(set.head(20))
    # print("shape:")
    # print(set.shape)

    # -- label encode non-numeric variables ------------------------------------------------------
    le_columns = [
        'WindGustDir',
        'WindDir9am',
        'WindDir3pm',
        'RainToday',
        'RainTomorrow'
    ]
    set[le_columns] = set[le_columns].apply(lambda x: pd.factorize(x)[0])

    # print("\n---- After label-encoding")
    # print("\nhead:")
    # print(set.head(20))
    # # print("\ntail:")
    # # print(set.tail())
    # print("shape:")
    # print(set.shape)

    # -- impute missing values -------------------------------------------------------------------
    # using mode under RainTomorrow
    if (type_set == 'train'):
        mode_rt = set['RainTomorrow'].mode()[0]
        # print(f'mode_rt for train: {mode_rt}')
    set['RainTomorrow'] = set['RainTomorrow'].replace(-1, mode_rt)

    # using mean under variables (except RainToday and RainTomorrow)
    imp_columns = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustDir',
        'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am',
        'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
        'Pressure3pm', 'Temp9am', 'Temp3pm'
    ]
    if (type_set == 'train'):
        mean = set[imp_columns].mean()
        mode_wgd = set['WindGustDir'].mode()[0]
        mode_wdn = set['WindDir9am'].mode()[0]
        mode_wdt = set['WindDir3pm'].mode()[0]
        # print(f'mean for train: {mean}')
        # print(f'mode_wgd for train: {mode_wgd}')
        # print(f'mode_wdn for train: {mode_wdn}')
        # print(f'mode_wdt for train: {mode_wdt}')
    set[imp_columns] = set[imp_columns].fillna(mean)

    # using mode under WindGustDir, WindDir9am, WindDir3pm to impute encoded NaN values (i.e., -1)
    set['WindGustDir'] = set['WindGustDir'].replace(-1, mode_wgd)
    set['WindDir9am'] = set['WindDir9am'].replace(-1, mode_wdn)
    set['WindDir3pm'] = set['WindDir3pm'].replace(-1, mode_wdt)

    # print("\n---- After imputing missing values")
    # print("\nhead:")
    # print(set.head(20))
    # # print("\ntail:")
    # # print(set.tail())
    # print("shape:")
    # print(set.shape)

    return set, mode_rt, mean, mode_wgd, mode_wdn, mode_wdt
    # --------------------------------------------------------------------------------------------

# call the function for data cleaning
cleaned_train_set, mode_rt, mean, mode_wgd, mode_wdn, mode_wdt = cleanData(train_set, mode_rt, mean, mode_wgd, mode_wdn, mode_wdt, 'train')
# print("\n---- After data cleaning for training set:")
# print(cleaned_train_set)


# detect and eliminate outliers using z-score
threshold = 3  # drop records greater than this value
cleaned_train_set = cleaned_train_set[(np.abs(stats.zscore(cleaned_train_set)) < threshold).all(axis=1)]

# print("\n---- After eliminating outliers")
# print("\nhead:")
# print(cleaned_train_set.head())
# # print("\ntail:")
# # print(cleaned_train_set.tail())
# print("shape:")
# print(cleaned_train_set.shape)


# split X_train and y_train
cleaned_X_train = cleaned_train_set.iloc[:,:-1]                   # all columns except the last, RainTomorrow
cleaned_y_train = cleaned_train_set['RainTomorrow'].to_frame()    # RainTomorrow
# print("X_train:")
# print(cleaned_X_train)
# print("y_train:")
# print(cleaned_y_train)


# == NAIVE BAYES =================================================================================
# Training ---------------------------------------------------------------------------------------
model = GaussianNB()
model.fit(cleaned_X_train.values, cleaned_y_train.values.ravel())

# Testing ----------------------------------------------------------------------------------------
# Data Preprocessing
# combine X_test and y_test for data cleaning
test_set = pd.concat([X_test,y_test], axis=1)

# call the function for data cleaning
# print(f'mode_rt for test: {mode_rt}')
# print(f'mean for test: {mean}')
# print(f'mode_wgd for test: {mode_wgd}')
# print(f'mode_wdn for test: {mode_wdn}')
# print(f'mode_wdt for test: {mode_wdt}')
cleaned_test_set, mode_rt, mean, mode_wgd, mode_wdn, mode_wdt = cleanData(test_set, mode_rt, mean, mode_wgd, mode_wdn, mode_wdt, 'test')

# split X_train and y_train
cleaned_X_test = cleaned_test_set.iloc[:,:-1]                   # all columns except the last, RainTomorrow
cleaned_y_test = cleaned_test_set['RainTomorrow'].to_frame()    # RainTomorrow

# print("\ncleaned_X_test head:")
# print(cleaned_X_test.head(40))
# print("\ncleaned_X_test tail:")
# print(cleaned_X_test.tail(40))

# predict
y_pred = model.predict(cleaned_X_test.values)

# display results
print("y_pred")
print(y_pred)

print("\ny_test")
print(cleaned_y_test)

# Evaluate Performance ---------------------------------------------------------------------------
cm = confusion_matrix(cleaned_y_test, y_pred)
print("\nConfusion matrix:")
print(cm)

cr = classification_report(cleaned_y_test, y_pred)
print("\nClassification report:")
print(cr)