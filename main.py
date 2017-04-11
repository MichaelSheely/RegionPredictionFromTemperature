import pandas as pd
import numpy as np
import os.path
from geopy.geocoders import Nominatim
from geopy.point import Point
from geopy.exc import GeocoderTimedOut
from geopy.exc import GeocoderServiceError

TEMPERATURES_FILE = 'data/USCityTemperaturesAfter1850.csv'
CITY_STATE_FILE = 'data/city_state.csv'
STATE_REGION_FILE = 'data/StatesAndRegions.csv'
FINAL_TEMPERATURES_FILE = 'data/labeled_data.csv' # TODO: Change all relevant files names to this constant.

def add_state(path=TEMPERATURES_FILE):
    temp = pd.read_csv(path, header=0)
    only_unique = temp.sort_values(['City']).drop_duplicates(subset=['City','Latitude', 'Longitude'])
    if not os.path.isfile('data/States.csv'):
        states = create_state_column(only_unique)
    else:
        states = pd.read_csv('data/States.csv', header=0)
    joined_table = pd.merge(temp, states, how='left', on=['City', 'Latitude', 'Longitude'])
    joined_table.to_csv('data/Temp.csv', index=False)

def create_state_column(us):
    geolocator = Nominatim()
    us = us.assign(State=np.nan)
    lat_ndx = us.columns.get_loc('Latitude')
    lon_ndx = us.columns.get_loc('Longitude')
    state_ndx = us.columns.get_loc('State')
    city_ndx = us.columns.get_loc('City')
    def find_state(row):
        lat = row[lat_ndx]
        lon = row[lon_ndx]
        city = row[city_ndx]
        # print city,lat, lon
        coord = Point(lat + ' ' + lon)
        try:
            location = geolocator.reverse(coord)
        except  (GeocoderTimedOut, GeocoderServiceError) as e:
            if city == 'Chicago':
                state = 'Illinois'
            elif city == 'Milwaukee':
                state = 'Wisconsin'
            else:
                print "location %s at %s %s timed out" %(city, lat, lon)
                state = np.nan
            return state
        # print location.raw
        if location.raw['address']['country_code'] != 'us':
            print "ERRRRRROOORRRRR"
            return
        try:
            state = location.raw['address']['state']
        except KeyError as e:
            if lat == '32.95N' and lon == '117.77W':
                state = 'California'
            elif city in ['Anaheim', 'Chula Vista', 'San Diego']:
                state = 'California'
            elif city == 'Brownsville':
                state = 'Texas'
            else:
                print "location %s at %s %s keyed out" %(city, lat, lon)
                state= np.nan
        return state
    state = us.apply(find_state, axis=1)
    us = us.assign(State = state)
    print state

    us['City', 'Latitude', 'Longitude', 'State'].to_csv('data/States.csv', index=False)
    return us

def filter_and_save_data(path='data/GlobalLandTemperaturesbyCity.csv', ignore_before=1850):
    out = pd.read_csv(path, header=0)
    us = out.loc[out['Country'] == 'United States']
    # lexicograpical comparison of strings makes this work for any 4 digit year
    us = us[us['dt'] > '1850']
    # delete the old index and Country code
    us.drop('Country', axis=1, inplace=True)
    us.reset_index(drop=True, inplace=True)
    us.to_csv(TEMPERATURES_FILE, index=False)
    return us

def merge_data(to_merge='data/Temp.csv', new_file='data/joined.csv'):
    state_region = pd.read_csv(STATE_REGION_FILE, header=0)
    temperatures = pd.read_csv(to_merge, header=0)
    joined_table = pd.merge(temperatures, state_region, how='left', on='State')
    joined_table.to_csv(new_file, index=False)


def city_country(raw_file='data/RawUSData.csv'):
    out = pd.read_csv(raw_file)
    keep = ['Name', 'Canonical Name']
    us = out[keep]
    us = us.assign(State = us['Canonical Name'].apply(get_state))
    us = us.rename(columns={'Name':'City'})
    us = us[['City', 'State']]
    us.to_csv(CITY_STATE_FILE, index=False)

def get_state(raw_string):
    return raw_string.split(',')[-2]

def main():
    if not os.path.isfile(TEMPERATURES_FILE): # TODO: add force make file
        filter_and_save_data()
    data = pd.read_csv(TEMPERATURES_FILE)
    if not os.path.isfile(CITY_STATE_FILE):
        city_country()
    cities = pd.read_csv(CITY_STATE_FILE)
    add_state()
    merge_data(to_merge='data/States.csv', new_file='data/CityRegions.csv')
    merge_data()

if __name__ == "__main__":
    main()
