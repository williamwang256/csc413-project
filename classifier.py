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

import datetime

print(datetime.datetime.now())

BASE = "/h/u6/c9/01/wangwi18/winter24/csc413/project/data/"
DATA_PATH = BASE + "/data/"
OUTPUT_PATH = BASE + "/output/"
DATASET_PATH = BASE + "/metadata/bird_dataset.csv"
METADATA_PATH = BASE + "/metadata/birdsong_metadata.csv"
LABELS_PATH = BASE + "/metadata/labels.csv"

# SPECTROGRAM_PATH = BASE + "/spectrograms/Class"
# SPEC_AUGMENT_PATH = BASE + "/augmentations/spec_augment/"
# TIME_SHIFT_PATH = BASE + "/augmentations/time_shift/"

SORTED_PATH = BASE + "/sorted/"

# use ImageFolder to load the data to Pytorch Dataset Format
dataset = ImageFolder(SORTED_PATH) # REPLACE WITH CORRECT PATH
print(len(dataset))
print(dataset.classes)
print(len(dataset.classes))

random_seed = 45
torch.manual_seed(random_seed)

img, label = dataset[0]

plt.imshow(img)
print(label, dataset.classes[label])




# make a custom Bird Spectrogram Dataset,
class BirdSpectrogramDataset(Dataset):
  def __init__(self, ds, transform=None):
    self.ds = ds
    self.transform = transform

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    img, label = self.ds[idx]

    if self.transform:
      img = self.transform(img)

    return img, label

# UPDATE THESE RATIOS ONCE WE HAVE THE FULL DATASET
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
print(len(train_set), len(val_set), len(test_set))



# define transformations here
transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# define dataset
train_ds = BirdSpectrogramDataset(train_set, transform)
val_ds = BirdSpectrogramDataset(val_set, transform)
test_ds = BirdSpectrogramDataset(test_set, transform)

for X, t in train_ds:
  plt.imshow(X.permute(1,2,0))
  print(X.shape)
  print(t)
  break



# create DataLoaders
batch_size = 64

train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size*2)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size*2)


# move data to GPU
def get_default_device():
  if torch.cuda.is_available():
    return torch.device('cuda:0')
  else:
    return torch.device('cpu')

device = get_default_device()
print(device)



# ResNet50
class BirdSpectrogramResNet50(nn.Module):
  def __init__(self):
    super().__init__()

    self.network = models.resnet50(weights='ResNet50_Weights.DEFAULT')

    # modify last layer
    num_ftrs = self.network.fc.in_features # fc is the fully connected last layer
    self.network.fc = nn.Linear(num_ftrs, 88) # MAKE SURE TO UPDATE THIS TO THE CORRECT NUMBER OF BIRDS
    nn.init.xavier_uniform_(self.network.fc.weight) # initialize weights

  def forward(self, xb):
    return self.network(xb)

model = BirdSpectrogramResNet50()




def accuracy(model, dl, device):
  model.to(device)
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for X, t in dl:
      X = X.to(device)
      t = t.to(device)

      z = model(X)
      pred = z.max(1, keepdim=True)[1] # index of max log-probability
      correct += pred.eq(t.view_as(pred)).sum().item()
      total += X.shape[0]

  return correct / total

def train_model(model,
                train_dl,
                val_dl,
                learning_rate=0.001,
                epochs=10,
                plot_every=10,
                plot=True):

  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=learning_rate)

  # for plotting
  iters, train_loss, train_acc, val_acc = [], [], [], []
  iter_count = 0 # count the number of iterations that has passed

  for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch+1, epochs))

    # put model in train mode

    for X, t in train_dl:
      X = X.to(device)
      t = t.to(device)

      model.train()
      z = model(X)
      loss = criterion(z, t)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      iter_count += 1
      if iter_count % plot_every == 0:
        loss = float(loss)
        t_acc = accuracy(model, train_dl, device)
        v_acc = accuracy(model, val_dl, device)

        print("Iter {}; Loss {}; Train Acc {}; Val Acc {}".format(iter_count, loss, t_acc, v_acc))

        iters.append(iter_count)
        train_loss.append(loss)
        train_acc.append(t_acc)
        val_acc.append(v_acc)

    print()

  # plot result
  if plot:
    plt.figure()
    plt.plot(iters[:len(train_loss)], train_loss)
    plt.title("Loss over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.figure()
    plt.plot(iters[:len(train_acc)], train_acc)
    plt.plot(iters[:len(val_acc)], val_acc)
    plt.title("Accuracy over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])



train_model(model, train_dl, val_dl, learning_rate=0.001, epochs=1, plot_every=1, plot=False)
print("DONE!")
print(datetime.datetime.now())