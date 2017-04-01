import pandas as pd
import os.path

TEMPERATURES_FILE = 'data/USCityTemperaturesAfter1850.csv'
CITY_STATE_FILE = 'data/city_state.csv'

def filter_and_save_data(path='data/GlobalLandTemperaturesbyCity.csv', ignore_before=1850):
    out = pd.read_csv(path, header=0)
    us = out.loc[out['Country'] == 'United States']
    # lexicograpical comparison of strings makes this work for any 4 digit year
    us = us[us['dt'] > '1850']  
    # delete the old index and Country code
    us.drop('Country', axis=1, inplace=True)
    us.reset_index(drop=True, inplace=True)
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
        filter_and_save_data()
    data = pd.read_csv(TEMPERATURES_FILE)
    if not os.path.isfile(CITY_STATE_FILE):
        city_country()
    cities = pd.read_csv(CITY_STATE_FILE)

if __name__ == "__main__":
    main()
