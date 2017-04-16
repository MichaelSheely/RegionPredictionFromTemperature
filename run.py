import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.transformers import FeatureAugmenter
from tsfresh.feature_extraction import MinimalFeatureExtractionSettings
from tsfresh.feature_extraction.settings import FeatureExtractionSettings
from tsfresh.utilities.dataframe_functions import impute
from sklearn.metrics import accuracy_score

cities_dict = {}
counter = 0
def number_cities(row):
    global counter
    if not row in cities_dict:
        cities_dict[row] = counter
        counter += 1
    return cities_dict[row]

def number_regions(row):
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


def split_data(df, city_regions):
    # Get the mapping from each unique city to each region
    orig_cities = city_regions['City']
    X_cities = orig_cities.apply(number_cities)
    y_regions = city_regions['Region']
    y_regions = y_regions.apply(number_regions)

    y_train, y_test = train_test_split(y_regions)
    df_train = df.loc[df.City.isin(y_train.index)]
    df_test = df.loc[df.City.isin(y_test.index)]
    X_train = pd.DataFrame(index=y_train.index)
    X_test = pd.DataFrame(index=y_test.index)
    train_names = orig_cities[y_train.index]
    test_names = orig_cities[y_test.index]
    train = {'df': df_train, 'X': X_train, 'y': y_train, 'city_names': train_names}
    test = {'df': df_test, 'X': X_test, 'y': y_test, 'city_names': test_names}
    return (train, test)

def run(filename='data/joined.csv', city_regions_file='data/CityRegions.csv'):
    df = pd.read_csv(filename, header=0)
    df.dropna(inplace=True)

    X_labels = ['City', 'dt', 'AverageTemperature']
    df = df[X_labels]
    df = df.dropna()
    df['City'] = df['City'].apply(number_cities)


    if city_regions_file == None:
        temp = [['Abiline', 'South'],['West Jordon', 'West' ], ['Yonkers', 'Northeast']]
        city_regions = pd.DataFrame(temp, columns=['City', 'Region'])
    else:
        city_regions = pd.read_csv(city_regions_file, header=0)

    print city_regions
    train, test = split_data(df, city_regions)

    feature_extraction_settings = FeatureExtractionSettings()
    feature_extraction_settings.IMPUTE = impute
    aug = FeatureAugmenter(feature_extraction_settings, column_id='City',
                    column_sort='dt', column_value='AverageTemperature',
                    timeseries_container=train['df'])
    output = aug.fit_transform(train['X'], train['y'])
    output['City_Name'] = train['city_names']
    output.to_csv('features_from_tsfresh.csv', index=False)

    # DecisionTreeClassifier(criterion='entropy')
    #pipeline = Pipeline([('augmenter', FeatureAugmenter(feature_extraction_settings, column_id='City', column_sort='dt', column_value='AverageTemperature')),
    #                ('classifier', DecisionTreeClassifier(criterion='entropy'))])
    #pipeline = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='City', column_sort='dt', column_value='AverageTemperature')),
    #                ('classifier', DecisionTreeClassifier(criterion='entropy'))])

    # for the fit on the train test set, we set the fresh__timeseries_container to `df_train`
    pipeline.set_params(augmenter__timeseries_container=train['df'])
    pipeline.fit(train['X'], train['y'])

    y_pred = pipeline.predict(train['X'])
    y_true = np.array(train['y'])
    print "train accuracy ", accuracy_score(y_true, y_pred)

    # for the predict on the test test set, we set the fresh__timeseries_container to `df_test`
    pipeline.set_params(augmenter__timeseries_container=test['df'])
    y_pred = pipeline.predict(test['X'])

    y_true = np.array(test['y'])

    print "test accuracy ", accuracy_score(y_true, y_pred)
    print "done"

TEMPERATURE_FILE = 'data/joined.csv'
test_file = 'data/testSet.csv'
if __name__ == '__main__':
    #run(test_file, city_regions_file=None)
    run()

