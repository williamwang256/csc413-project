import os
import torch
import pandas as pd
import numpy as np
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from collections import OrderedDict
import subprocess

BASE = "/h/u6/c9/01/wangwi18/winter24/csc413/project/data/"
DATA_PATH = BASE + "/data/"
OUTPUT_PATH = BASE + "/output/"
DATASET_PATH = BASE + "/metadata/bird_dataset.csv"
METADATA_PATH = BASE + "/metadata/birdsong_metadata.csv"
LABELS_PATH = BASE + "/metadata/labels.csv"

SPECTROGRAM_PATH = BASE + "/small/"
# SPEC_AUGMENT_PATH = BASE + "/augmentations/spec_augment/"
# TIME_SHIFT_PATH = BASE + "/augmentations/time_shift/"

SORTED_PATH = BASE + "/sorted_small/"

df = pd.read_csv(DATASET_PATH, header=None)
# print(df.iat[51, 0]) # should be eurasian golden oriole

dataset = ImageFolder(SPECTROGRAM_PATH)

for img, index in dataset.imgs:
  i = int(img[-6:-4].replace("_", "")) # get bird index
  bird_name = df.iat[i, 0]

  # make directory
  path = SORTED_PATH + bird_name
  os.makedirs(path, exist_ok=True)

  path = "\"" + path+"\""
  img = "\"" + img+"\""
  
  # add image to directory
  command = f"cp -r {img} {path}"
  subprocess.Popen(command, shell=True)