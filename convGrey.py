import os
from PIL import Image

for filename in os.listdir('./output/'):
    img = Image.open('./output/' + filename).convert('L')
    img.save('./greyscale/' + filename)
