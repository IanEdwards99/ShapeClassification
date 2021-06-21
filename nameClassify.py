import os
import csv
from PIL import Image

path = './Extra Data/triangles/'
savepath = './preprocessing/'
classification = 'triangle_0'
i = 1
for filename in os.listdir(path):
    img = Image.open(path + filename)
    img.save(savepath + classification + str(i) + ".png")
    i += 1
