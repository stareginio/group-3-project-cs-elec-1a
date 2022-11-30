import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dataset_csv = "weatherAUS"
dataset = pd.read_csv(f"{dataset_csv}.csv")

print("Before data cleaning")
print(dataset.head())
print(dataset.shape)

# drop variables that are irrelevant and has many NA values
to_drop = [
    'Evaporation',
    'Sunshine',
    'Cloud9am',
    'Cloud3pm',
    'Location',
    'Date'
]
dataset.drop(to_drop, inplace=True, axis=1)

# drop records with NA values under RainToday
dataset.dropna(subset=['RainToday'], inplace=True)

print("\nAfter data cleaning")
print(dataset.head())
print(dataset.shape)