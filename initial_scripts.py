import pandas as pd

out = pd.read_csv("data/GlobalLandTemperaturesbyCity.csv", header=0)
us = out.loc[out['Country'] == 'United States']
print "with nans ", us.shape
us = us.dropna()
print "without nans ", us.shape
print "unique ", len(us['City'].value_counts())

out = pd.read_csv("data/GlobalLandTemperaturesbyMajorCity.csv", header=0)
us = out.loc[out['Country'] == 'United States']
print "with nans ", us.shape
us = us.dropna()
print "without nans ", us.shape
print us['City'].value_counts()
print "unique ", len(us['City'].value_counts())
