#Author: Ian Edwards

import os
import csv

shapeClass = {
    "Circle" : 0,
    "Triangle" : 1,
    "Square" : 2,
    "Pentagon" : 3,
    "Star" : 4,
    "Hexagon" : 5,
    "Heptagon" : 6,
    "Octagon" : 7,
    "Nonagon" : 8
}

f = open('shape_data.csv', 'w')
writer = csv.writer(f)

for filename in os.listdir('./output/'):
    shapeName = filename[:filename.find('_')]
    shapeNr = shapeClass[shapeName]
    row = [filename, shapeNr]
    writer.writerow(row)

f.close()
