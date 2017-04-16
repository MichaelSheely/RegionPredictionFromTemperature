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
    key = (row[0],row[1])
    if not key in cities_dict:
        print key
        cities_dict[key] = counter
        counter += 1
    return cities_dict[key]

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
    orig_cities = city_regions[['City','State']]
    print "Total cities ", len(orig_cities)
    y_regions = city_regions['Region']
    y_regions = y_regions.apply(number_regions)

    y_train, y_test = train_test_split(y_regions)
    df_train = df.loc[df.CityIndex.isin(y_train.index)]
    df_test = df.loc[df.CityIndex.isin(y_test.index)]
    X_train = pd.DataFrame(index=y_train.index)
    X_test = pd.DataFrame(index=y_test.index)
    print len(orig_cities)
    print len(y_train.index)
    print orig_cities
    print y_train.index
    print max(y_train.index)
    train_names = orig_cities.iloc[y_train.index]
    test_names = orig_cities.iloc[y_test.index]

    train = {'df': df_train, 'X': X_train, 'y': y_train, 'city_names': train_names}
    test = {'df': df_test, 'X': X_test, 'y': y_test, 'city_names': test_names}
    return (train, test)

def run(filename='data/clean_data.csv', city_regions_file='data/CityRegions.csv'):
    df = pd.read_csv(filename, header=0)
    df.dropna(inplace=True)

    X_labels = ['City', 'State', 'dt', 'AverageTemperature', 'CityIndex']
    df = df[X_labels]
    df = df.dropna()
    city_state = df[['City', 'State']]
    # Sadness because multiple cities with same name.......
    #hi = city_state.apply(number_cities, axis=1)
    #df['CityIndex'] = city_state.apply(number_cities, axis=1)
    #df.to_csv('data/clean_data.csv', index=False)

    if city_regions_file == None:
        temp = [['Abiline', 'South'],['West Jordon', 'West' ], ['Yonkers', 'Northeast']]
        city_regions = pd.DataFrame(temp, columns=['City','Region'])
    else:
        city_regions = pd.read_csv(city_regions_file, header=0).reset_index(drop=True)
        

    train, test = split_data(df, city_regions)
    """
    blah = train['city_names']
    print blah
    indices = blah[blah.isin(['Yonkers','Worcester','Winston Salem','Windsor' ,'Wichita' ,'Westminster' ,'West Valley City' ,'West Jordan'])].index
    print indices
    print train['df'][train['df']['City'].isin(indices)]
    print train['df']['City']
    df = train['df'][train['df']['City'].isin(indices)]
    X = train['X'][indices]
    y = train['y'][indices]
    city_names = train['city_names'][indices]
    """

    
    feature_extraction_settings = FeatureExtractionSettings()
    feature_extraction_settings.IMPUTE = impute
    aug = FeatureAugmenter(feature_extraction_settings, column_id='CityIndex',
                    column_sort='dt', column_value='AverageTemperature',
                    timeseries_container=train['df'])
    output = aug.fit_transform(train['X'], train['y'])
    output['City_Name'] = train['city_names']
    output.to_csv('data/features_from_tsfresh.csv', index=False)

    # DecisionTreeClassifier(criterion='entropy')
    #pipeline = Pipeline([('augmenter', FeatureAugmenter(feature_extraction_settings, column_id='City', column_sort='dt', column_value='AverageTemperature')),
    #                ('classifier', DecisionTreeClassifier(criterion='entropy'))])
    #pipeline = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='City', column_sort='dt', column_value='AverageTemperature')),
    #                ('classifier', DecisionTreeClassifier(criterion='entropy'))])

    # for the fit on the train test set, we set the fresh__timeseries_container to `df_train`
    """
    pipeline.set_params(augmenter__timeseries_container=train['df'])
    pipeline.fit(train['X'], train['y'])

    y_pred = pipeline.predict(train['X'])
    y_true = np.array(train['y'])
    print "train accuracy ", accuracy_score(y_true, y_pred)
    """
    # for the predict on the test test set, we set the fresh__timeseries_container to `df_test`
    """
    pipeline.set_params(augmenter__timeseries_container=test['df'])
    y_pred = pipeline.predict(test['X'])

    y_true = np.array(test['y'])
    """

    print "test accuracy ", accuracy_score(y_true, y_pred)
    print "done"

TEMPERATURE_FILE = 'data/joined.csv'
test_file = 'data/testSet.csv'
if __name__ == '__main__':
    #run(test_file, city_regions_file=None)
    run()

