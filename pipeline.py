import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# TODO: data should be formated with all the cities -> regions
city_regions = pd.read_csv('data/CityRegions.csv', header=0)
X_cities = city_regions['City']
y_cities = city_regions['Region']
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


pipeline = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='AverageTemperature', column_sort='dt')),
        ('classifier', DecisionTreeClassifier())])
pipeline.set_params(augmenter__timeseries_container=df_ts)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_train)
y_true = np.array(y_train)

# ADD METRICS HERE

