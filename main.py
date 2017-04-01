import pandas as pd
import os.path

TEMPERATURES_FILE = 'data/USCityTemperaturesAfter1850.csv'
CITY_STATE_FILE = 'data/city_state.csv'

def load_data(path='data/GlobalLandTemperaturesbyCity.csv', ignore_before=1850):
    out = pd.read_csv(path, header=0)
    us = out.loc[out['Country'] == 'United States']
    us = us.loc[us['dt'] > 1850]
    us.to_csv(TEMPERATURES_FILE)
    return us

def city_country(raw_file='data/RawUSData.csv'):
    out = pd.read_csv(raw_file)
    keep = ['Name', 'Canonical Name']
    us = out[keep]
    us = us.assign(State = us['Canonical Name'].apply(get_state))
    us = us.rename(columns={'Name':'City'})
    us = us[['City', 'State']]
    us.to_csv(CITY_STATE_FILE)

def get_state(raw_string):
    return raw_string.split(',')[-2]

def main():
    if not os.path.isfile(TEMPERATURES_FILE): # TODO: add force make file
        load_data()
    data = pd.read_csv(TEMPERATURES_FILE)
    if not os.path.isfile(CITY_STATE_FILE):
        city_country()
    cities = pd.read_csv(CITY_STATE_FILE)

if __name__ == "__main__":
    main()
