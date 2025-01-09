import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the dataset
dataframe = pd.read_csv('data.csv')
print(dataframe.head(5))

df = dataframe.copy()
df.dropna(inplace=True, axis=1)
print(df.isnull().sum())
df.drop('id', axis=1, inplace=True)