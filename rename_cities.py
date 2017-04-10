# script to change the names of duplicate cities to include the state they are in
import csv

cityMap = {("100.53W", "32.95N"): "Abilene_2", ("74.56W", "40.99N"):"Yonkers_2"}

with open('data/USCityTemperaturesAfter1850.csv', mode='r') as csv_in:
    reader = csv.reader(csv_in)
    with open('data/USCityTemperaturesAfter1850UniqueCities.csv', mode='wb') as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(reader.next())  # pass over the first row, which is just schema
        for row in reader:
            print row
            coordinate = (row[6], row[5])
            if coordinate in cityMap:
                row[4] = cityMap[coordinate]
            writer.writerow(row)


