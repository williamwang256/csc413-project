import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from config import *

ATTN_MAP = "attn_map1.jpg"
MODEL = "google/vit-base-patch16-224-in21k"
SAVE_DIR = "vit-base-birds"

df = pd.read_csv(METADATA_PATH, header=0)

# Uses the model to make a prediction for the given image
def predict(img, model, processor):
  inputs = processor(images=img, return_tensors="pt")
  idx = int(torch.argmax(model(**inputs).logits, dim=1)[0])
  return sorted(list(set(df["english_cname"])))[idx]

# Obtains the attention map for the given image
def get_attention_map(model, processor, img, get_mask=False):
  model = model.cpu()   # this computation will be done on CPU

  # Pass the input through the model
  inputs = processor(images=img, return_tensors="pt")
  output = model(**inputs, output_attentions=True)

  # Access attention maps
  att_mat = torch.stack(output.attentions).squeeze(1)

  # Average the attention weights across all heads.
  att_mat = torch.mean(att_mat, dim=1)

  # To account for residual connections, we add an identity matrix to the
  # attention matrix and re-normalize the weights.
  residual_att = torch.eye(att_mat.size(1))
  aug_att_mat = att_mat + residual_att
  aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

  # Recursively multiply the weight matrices
  joint_attentions = torch.zeros(aug_att_mat.size())
  joint_attentions[0] = aug_att_mat[0]

  for n in range(1, aug_att_mat.size(0)):
    joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

  v = joint_attentions[-1]
  grid_size = int(np.sqrt(aug_att_mat.size(-1)))
  mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
  if get_mask:
    result = cv2.resize(mask / mask.max(), img.size)
  else:        
    mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
    result = (mask * img).astype("uint8")
  
  return result

# Plots the attention map alongside the original image
def plot_attention_map(original_img, att_map):
  _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
  ax1.set_title("Original")
  ax2.set_title("Attention Map Last Layer")
  _ = ax1.imshow(original_img)
  _ = ax2.imshow(att_map)
  plt.axis("off")
  plt.savefig(ATTN_MAP)


if __name__ == "__main__":

  model = ViTForImageClassification.from_pretrained(SAVE_DIR)
  processor = ViTImageProcessor.from_pretrained(MODEL)

  img = Image.open(SPECTROGRAM_PATH + "/Eurasian Skylark/xc123169_038.jpg")
  prediction = predict(img, model, processor)
  attn_map = get_attention_map(model, processor, img, True)
  plot_attention_map(img, attn_map)
  print(prediction)