#Author: Ian Edwards
#Description: Converts images to greyscale and resizes to 200x200.

import os
from PIL import Image
import sys
import argparse

width = 200 #Chosen images are all 200x200
height = 200
text = 'Enter the directory to convert to greyscale.'

parser = argparse.ArgumentParser(description=text) #setup argument parser.
parser.add_argument('path')
args = parser.parse_args()

for filename in os.listdir(args.path):
    img = Image.open(args.path + filename).convert('L')
    img = img.resize((width, height), Image.ANTIALIAS)
    img.save(args.path + filename)
