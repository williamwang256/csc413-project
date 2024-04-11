import torch
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from BirdSpectrogramDataset import BirdSpectrogramDataset
from BirdSpectrogramResNet50 import BirdSpectrogramResNet50
from config import *

# move data to GPU
def get_default_device():
  if torch.cuda.is_available():
    return torch.device('cuda:0')
  else:
    return torch.device('cpu')

if __name__ == "__main__":
  device = get_default_device()
  print("Device: ", device)

  random_seed = 45
  torch.manual_seed(random_seed)

  # use ImageFolder to load the data to Pytorch Dataset Format
  dataset = ImageFolder(SORTED_PATH) # REPLACE WITH CORRECT PATH
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

  # define dataset
  train_ds = BirdSpectrogramDataset(train_set, transform)
  val_ds   = BirdSpectrogramDataset(val_set, transform)
  test_ds  = BirdSpectrogramDataset(test_set, transform)

  # create DataLoaders
  batch_size = 64
  train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
  val_dl   = torch.utils.data.DataLoader(val_ds, batch_size * 2)
  test_dl  = torch.utils.data.DataLoader(test_ds, batch_size * 2)

  model = BirdSpectrogramResNet50(len(dataset.classes))

  # Train the model
  BirdSpectrogramResNet50.train_model(model, device, train_dl, val_dl, learning_rate=0.001, epochs=10, plot_every=1, plot=False)

  # Final test accuracy
  print("test acc {}".format(BirdSpectrogramResNet50.accuracy(model, test_dl, device)))