import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.transformers import FeatureAugmenter
from tsfresh.feature_extraction import MinimalFeatureExtractionSettings
from tsfresh.feature_extraction.settings import FeatureExtractionSettings
from tsfresh.utilities.dataframe_functions import impute

# TODO: data should be formated with all the cities -> regions
city_regions = pd.read_csv('data/CityRegions.csv', header=0)
X_cities = city_regions['City']
y_regions = city_regions['Region']


"""
train_cities, test_cities = train_test_split(X_cities, y_cities, stratify=y_cities)

data = pd.read_csv('data/Temp.csv', header=0)
train_ndx = data[data['City'].isin(train_cities)].index.tolist()
test_ndx = data[data['City'].isin(test_cities)].index.tolist()

X_labels = ['AverageTemperature', 'dt']
y_label = 'Region'
X_train = data[train_ndx][X_labels]
y_train = data[train_ndx][y_label]
X_test = data[test_ndx][X_labels]
y_test = data[test_ndx][y_label]
"""

from tsfresh.examples import load_robot_execution_failures
import tsfresh.examples
tsfresh.examples.robot_execution_failures.download_robot_execution_failures()


def change(row):
    if row == 'Northeast':
        return 0
    elif row == 'Midwest':
        return 1
    elif row == 'West':
        return 2
    elif row == 'South':
        return 3
    else:
        print "AAAAAAAAAA", row
    return None

cities = {}
counter = 0
def relabel(row):
    global counter
    if not row in cities:
        cities[row] = counter
        counter += 1
    return cities[row]

#TEMPERATURE_FILE = 'data/joined.csv'
TEMPERATURE_FILE = 'data/testSet.csv'
df = pd.read_csv(TEMPERATURE_FILE, header=0)
df.dropna(inplace=True)

X_labels = ['City', 'dt', 'AverageTemperature']
X_train = df[X_labels]
X_train = X_train.dropna()
X_train['City'] = X_train['City'].apply(relabel)

testArray = [['Abiline', 'South'],['West Jordon', 'West' ], ['Yonkers', 'Northeast']]
test_cities_regions = pd.DataFrame(testArray, columns=['City', 'Region'])
y_regions = test_cities_regions['Region']
y_regions = y_regions.apply(change)
X_empty = pd.DataFrame(index=y_regions.index)

df_ts, y = load_robot_execution_failures()
X = pd.DataFrame(index=y.index)
print X.shape
print X_empty.shape

"""
pipeline = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
                ('classifier', DecisionTreeClassifier())])
pipeline.set_params(augmenter__timeseries_container=df_ts)
pipeline.fit(X, y)
quit()

"""


print y_regions.shape
feature_extraction_settings = FeatureExtractionSettings()
feature_extraction_settings.IMPUTE = impute
pipeline = Pipeline([('augmenter', FeatureAugmenter(feature_extraction_settings, column_id='City', column_sort='dt', column_value='AverageTemperature')),
                ('classifier', DecisionTreeClassifier(criterion='entropy'))])

pipeline.set_params(augmenter__timeseries_container=X_train)
pipeline.fit(X_empty, y_regions)

"""
aug = RelevantFeatureAugmenter(column_id='City', column_sort='dt', column_value="AverageTemperature", timeseries_container=X_train)
new_X = aug.fit_transform(X_empty, y_regions)

clf = DecisionTreeClassifier(criterion='entropy')
"""


y_pred = pipeline.predict(X_empty)
print X_empty
y_true = np.array(y_regions)
from sklearn.metrics import accuracy_score

print y_pred

print "accuracy ", accuracy_score(y_true, y_pred)
print "done"

# ADD METRICS HERE

