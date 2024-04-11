from torch.utils.data import Dataset

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