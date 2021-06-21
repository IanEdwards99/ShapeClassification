import os
import csv
from PIL import Image

path = './preprocessing/'
savepath = './preprocessing/'
classification = 'triangle_00'
i = 1
for filename in os.listdir(path):
    img = Image.open(path + filename)
    img.save(savepath + classification + str(i) + ".png")
    i += 1
