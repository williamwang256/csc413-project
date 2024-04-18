import matplotlib.pyplot as plt
import numpy as np
from random import randint, uniform
import torch
import sys
import torch.nn as nn
from torch.utils.data import random_split, Dataset
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torchvision.transforms as transforms

from config import *

# ResNet50
class BirdSpectrogramResNet50(nn.Module):
  def __init__(self, num_species):
    super().__init__()

    self.network = models.resnet50(weights='ResNet50_Weights.DEFAULT')

    # modify last layer
    num_ftrs = self.network.fc.in_features # fc is the fully connected last layer
    self.network.fc = nn.Linear(num_ftrs, num_species) # MAKE SURE TO UPDATE THIS TO THE CORRECT NUMBER OF BIRDS
    nn.init.xavier_uniform_(self.network.fc.weight) # initialize weights

  def forward(self, xb):
    return self.network(xb)

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
                device,
                train_ds,
                val_ds,
                learning_rate=0.001,
                batch_size=64,
                epochs=10,
                plot_every=10,
                plot=True):

  train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
  val_dl   = torch.utils.data.DataLoader(val_ds, 256)

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
      if plot and iter_count % plot_every == 0:
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
    plt.savefig("loss.png")

    plt.figure()
    plt.plot(iters[:len(train_acc)], train_acc)
    plt.plot(iters[:len(val_acc)], val_acc)
    plt.title("Accuracy over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])
    plt.savefig("acc.png")

# Performs time and frequency masking as a form of data augmentation on the spectrogram.
# Referenced from: https://www.kaggle.com/code/CVxTz/audio-data-augmentation/notebook
# NOTE: we chose not to apply this transformation to the final model, but we leave the code here for completeness.
def spec_augment(original_melspec, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
  augmented_melspec = original_melspec.clone()
  # print(augmented_melspec.shape)
  _, all_frames_num, all_freqs_num = augmented_melspec.shape

  # Frequency masking
  freq_percentage = uniform(0.0, freq_masking_max_percentage)
  num_freqs_to_mask = int(freq_percentage * all_freqs_num)
  f0 = int(np.random.uniform(low = 0.0, high = (all_freqs_num - num_freqs_to_mask)))
  augmented_melspec[:, :, f0:(f0 + num_freqs_to_mask)] = 0

  # Time masking
  time_percentage = uniform(0.0, time_masking_max_percentage)
  num_frames_to_mask = int(time_percentage * all_frames_num)
  t0 = int(np.random.uniform(low = 0.0, high = (all_frames_num - num_frames_to_mask)))
  augmented_melspec[:, t0:(t0 + num_frames_to_mask), :] = 0

  return augmented_melspec

if __name__ == "__main__":
  # get arg
  n = len(sys.argv)
  if (n == 1):
    print("Usage: classifier.py <no aug (1) / yes aug (0)>")
    exit(0)
  no_aug = int(sys.argv[1])

  device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
  print("Device: ", device)

  random_seed = 45
  torch.manual_seed(random_seed)

  # use ImageFolder to load the data to Pytorch Dataset Format
  dataset = ImageFolder(SPECTROGRAM_PATH) # REPLACE WITH CORRECT PATH
  print("Number of samples: ", len(dataset))
  # print(dataset.classes)
  print("Number of classes: ", len(dataset.classes))

  # UPDATE THESE RATIOS ONCE WE HAVE THE FULL DATASET
  train_size = int(0.6 * len(dataset))
  val_size   = int(0.2 * len(dataset))
  test_size  = len(dataset) - train_size - val_size

  train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
  print("Train set size: ", len(train_set))
  print("Validation set size: ", len(val_set))
  print("Test set size: ", len(test_set))

  # define transformations here
  transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((224, 224)),
    transforms.ToTensor()
  ])
  transform_aug = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.roll(x, randint(1, 10000), dims=2))
    # transforms.Lambda(lambda x: spec_augment(x))
  ])

  # define dataset
  if (no_aug):
    train_ds = BirdSpectrogramDataset(train_set, transform)
  else:
    train_ds = BirdSpectrogramDataset(train_set, transform_aug)

  val_ds   = BirdSpectrogramDataset(val_set, transform)
  test_ds  = BirdSpectrogramDataset(test_set, transform)

  model = BirdSpectrogramResNet50(len(dataset.classes))

  # Train the model
  train_model(model, device, train_ds, val_ds, learning_rate=0.001, epochs=10, plot_every=10, plot=False)

  # Final test accuracy
  test_dl  = torch.utils.data.DataLoader(test_ds, 256)
  print("test acc {}".format(accuracy(model, test_dl, device)))