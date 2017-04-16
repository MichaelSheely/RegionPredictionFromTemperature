import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('data/joined.csv', header=0)
df.dropna(inplace=True)

fresno_frame = df[df['City'] == 'Fresno']
ts = Series(fresno_frame['AverageTemperature'], index=fresno_frame['dt'])
ts.cumsum()
ts.plot()




