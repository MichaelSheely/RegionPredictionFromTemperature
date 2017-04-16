import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

df = pd.read_csv('data/joined.csv', header=0)
df.dropna(inplace=True)

city_name = ""
if len(sys.argv) < 2:
    city_name = "Fresno"
else:
    city_name = sys.argv[1]

city_frame = df[df['City'] == city_name]
print city_frame['AverageTemperature']
ts = city_frame['AverageTemperature']
print ts
ts.index = city_frame['dt']
#ts = pd.Series(city_frame['AverageTemperature'], index=city_frame['dt'])
print ts

ts.cumsum()
ts.plot()

plt.show()



