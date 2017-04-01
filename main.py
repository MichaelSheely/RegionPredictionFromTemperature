import pandas as pd

def load_data(path='data/GlobalLandTemperaturesbyCity.csv', ignore_before=1850):
    out = pd.read_csv("data/GlobalLandTemperaturesbyCity.csv", header=0)
    us = out.loc[out['Country'] == 'United States']
    us = us.loc[us['dt'] > 1850]
    us.to_csv('data/USCityTemperaturesAfter1850.csv')
    return us


def main():
    data = pd.read_csv('data/USCityTemperaturesAfter1850.csv')

if __name__ == "__main__":
    main()
