import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pygame
import time
import random

from matplotlib.markers import CARETRIGHT

warnings.filterwarnings('ignore')

# Load the data set
df = pd.read_csv('kaggle/traffic.csv')

## Display head of the data set.
print(df.head(5))

## Shape of data set
print(df.shape)

## Information of the data set.
print(df.info())

## Null values check
null_values = df.isnull().sum()
print(null_values)

## Creating box plot by keeping vehicles in mind to detect outliers
plt.figure(figsize = (8,8))
sns.boxplot(x = df['Junction'], y = df['Vehicles'], data = df)

plt.xlabel('Junction data')
plt.ylabel('Vehicles data')
plt.tight_layout()
plt.show()

df.groupby('Junction')['Vehicles'].describe()