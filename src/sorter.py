import os
import subprocess

import pandas as pd
from torchvision.datasets import ImageFolder

from config import *

df = pd.read_csv(DATASET_PATH, header=None)
# print(df.iat[51, 0]) # should be eurasian golden oriole

dataset = ImageFolder("data/small_jpg/")

for img, index in dataset.imgs:
  i = int(img[-6:-4].replace("_", "")) # get bird index
  bird_name = df.iat[i, 0]

  # make directory
  path = "data/small_sorted_jpg/" + bird_name
  os.makedirs(path, exist_ok=True)

  path = "\"" + path+"\""
  img = "\"" + img+"\""
  
  # add image to directory
  command = f"cp -r {img} {path}"
  subprocess.Popen(command, shell=True)