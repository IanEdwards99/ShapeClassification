#Reference: 
#https://www.youtube.com/watch?v=ZoZHd0Zm3RY

import os
import pandas as pd
from PIL import Image
from numpy import asarray
import torch
from torch.utils.data import Dataset
from skimage import io

class shapeDataset(Dataset):
    def __init__(self, csvfile, rootdir, transform = None):
        self.annotations = pd.read_csv(csvfile)
        self.rootdir = rootdir
        self.transform = transform

    def __len__(self):
        return len(self.annotations) #54322

    def __getitem__(self, index):
        path = os.path.join(self.rootdir, self.annotations.iloc[index, 0])
        img = Image.open(path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            img = self.transform(img)
        
        return (img, y_label)