import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
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

    #y_other, y_val = train_test_split(y_regions, random_state=5, test_size=0.1, stratify=y_regions)
    #y_train, y_test = train_test_split(y_other, random_state=10, stratify=y_other)
    y_train, y_test = train_test_split(y_regions, random_state=10)
    #print "Validation cities: ", len(y_val)
    #print orig_cities.iloc[y_val.index]
    print "Num train: ", len(y_train)
    print "Num test: ", len(y_test)
    df_train = df.loc[df.CityIndex.isin(y_train.index)]
    df_test = df.loc[df.CityIndex.isin(y_test.index)]
    X_train = pd.DataFrame(index=y_train.index)
    X_test = pd.DataFrame(index=y_test.index)
    train_names = orig_cities.iloc[y_train.index]
    test_names = orig_cities.iloc[y_test.index]

    train = {'df': df_train, 'X': X_train, 'y': y_train, 'city_names': train_names}
    test = {'df': df_test, 'X': X_test, 'y': y_test, 'city_names': test_names}
    return (train, test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def run(filename='data/clean_data.csv', city_regions_file='data/CityRegions.csv', load_from_file=True):
    df = pd.read_csv(filename, header=0)
    df.dropna(inplace=True)

    X_labels = ['City', 'State', 'dt', 'AverageTemperature', 'CityIndex']
    df = df[X_labels]
    df = df.dropna()
    city_state = df[['City', 'State']]
    # Sadness because multiple cities with same name.......
    #df['CityIndex'] = city_state.apply(number_cities, axis=1)
    #df.to_csv('data/clean_data.csv', index=False)

    if city_regions_file == None:
        temp = [['Abiline', 'Texas','South'],['West Jordon', 'Utah', 'West' ], ['Yonkers','New York', 'Northeast']]
        city_regions = pd.DataFrame(temp, columns=['City', 'State','Region'])
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
    """
    aug = FeatureAugmenter(feature_extraction_settings, column_id='CityIndex',
                    column_sort='dt', column_value='AverageTemperature',
                    timeseries_container=train['df'])
    output = aug.fit_transform(train['X'], train['y'])
    output['City_Name'] = train['city_names']
    output.to_csv('data/features_from_tsfresh.csv', index=False)
"""

    if load_from_file:
        pipeline = joblib.load('./model.joblib.pkl')
    else:
        clf = DecisionTreeClassifier(criterion='entropy')
        feat_extractor = FeatureAugmenter(feature_extraction_settings,
                                          column_id='CityIndex', column_sort='dt', column_value='AverageTemperature')
        pipeline = Pipeline([('augmenter', feat_extractor),
                        ('classifier', clf)])
        #pipeline = Pipeline([('augmenter', RelevantFeatureAugmenter(column_id='CityIndex', column_sort='dt', column_value='AverageTemperature')),
        #                ('classifier', DecisionTreeClassifier(criterion='entropy'))])

        # for the fit on the train test set, we set the fresh__timeseries_container to `df_train`
        feat_extractor.set_timeseries_container(train['df'])
        output = feat_extractor.fit_transform(train['X'],train['y'])
        clf.fit(output, train['y'])
        # pipeline.set_params(augmenter__timeseries_container=train['df'])
        # pipeline.fit(train['X'], train['y'])

        y_pred = pipeline.predict(train['X'])
        y_true = np.array(train['y'])
        print "train accuracy ", accuracy_score(y_true, y_pred)
        cm_train = confusion_matrix(y_true, y_pred)
        print "Confusion matrix for training", cm_train
        # for the predict on the test test set, we set the fresh__timeseries_container to `df_test`
        pipeline.set_params(augmenter__timeseries_container=test['df'])
        joblib.dump(pipeline, './model.joblib.pkl')
    #### ENDIF

    y_pred = pipeline.predict(test['X'])

    y_true = np.array(test['y'])

    print "test accuracy ", accuracy_score(y_true, y_pred)
    cm_test = confusion_matrix(y_true, y_pred)
    print "Confusion matrix for testing", cm_test
    print "done"

    class_names = ['Northeast', 'Midwest', 'West', 'South']
    plot_confusion_matrix(cm_train, class_names)
    plt.savefig('train_cm.png')
    plt.hold(False)
    plot_confusion_matrix(cm_test, class_names)
    plt.savefig('test_cm.png')

    if not load_from_file:
        features = output.columns.values
        importances = clf.feature_importances_
        top_n = 20
        ndx = np.argsort(importances)[::-1]
        sorted_features = features[ndx][:20]
        sorted_importances = importances[ndx][:20]
        print '%80s   %s' %('Feature', 'Importance')
        for f, i in zip(sorted_features, sorted_importances):
            print '%80s   %.2f' % (f[20:], i)


TEMPERATURE_FILE = 'data/joined.csv'
test_file = 'data/testSet.csv'
if __name__ == '__main__':
    #run(test_file, city_regions_file=None, load_from_file=False)
    run(load_from_file=False)

