import tsfresh
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

#TEMPERATURES_FILE = 'data/USCityTemperaturesAfter1850.csv'
TEMPERATURES_FILE = 'data/testSet.csv'
df = pd.read_csv(TEMPERATURES_FILE, header=0)
# print pd.isnull(df).sum() > 0
df.dropna(inplace=True)
# impute(df)
extracted_features = tsfresh.extract_features(df, column_id="City",
                        column_sort="dt", column_value="AverageTemperature")
impute(extracted_features)
print extracted_features.describe()


