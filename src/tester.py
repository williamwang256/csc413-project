import torch
import pandas as pd
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from config import *

MODEL = "google/vit-base-patch16-224-in21k"
SAVE_DIR = "vit-base-birds"

df = pd.read_csv(METADATA_PATH, header=0)

def predict(img, model, processor):
  inputs = processor(images=img, return_tensors="pt")
  idx = int(torch.argmax(model(**inputs).logits, dim=1)[0])
  return sorted(list(set(df["english_cname"])))[idx]


model = ViTForImageClassification.from_pretrained(SAVE_DIR)
processor = ViTImageProcessor.from_pretrained(MODEL)

img = Image.open(SPECTROGRAM_PATH + "/Common Chiffchaff/xc71748_008.jpg")
print(predict(img, model, processor))