# https://huggingface.co/blog/fine-tune-vit
# https://huggingface.co/docs/datasets/en/create_datasethttps://huggingface.co/docs/datasets/en/create_dataset
# https://huggingface.co/docs/datasets/en/quickstart#vision

import torch
import numpy as np
from random import randint
from evaluate import load
from datasets import load_dataset, DatasetDict
from transformers import Trainer, TrainingArguments, ViTForImageClassification, ViTImageProcessor, ViTFeatureExtractor
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from config import *

# We will fine-tune the following pre-trained model
MODEL = "google/vit-base-patch16-224-in21k"

# Load the dataset, and split into train, test, and validation sets
# Use the split: 60% train, 20% validation, 20% test
full = load_dataset("imagefolder", data_dir=SPECTROGRAM_PATH, split="train")
train_testvalid = full.train_test_split(test_size=0.4)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
ds = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

# Load the pre-trained model's preprocessor
processor = ViTImageProcessor.from_pretrained(MODEL)

# A transformation which applies the above processor and some data
# augmentation steps (time shifting)
def transform(example_batch):
  inputs = processor([x for x in example_batch["image"]], return_tensors="pt")
  inputs["pixel_values"] = torch.roll(inputs["pixel_values"], randint(1, 10000), dims=3)
  inputs["label"] = example_batch["label"]
  return inputs

# Apply the transformation to the dataset. Note: the transformation is applied
# to examples only as they are indexed
prepared_ds = ds.with_transform(transform)

# Create our model
labels = ds["train"].features["label"].names
model = ViTForImageClassification.from_pretrained(
  MODEL,
  num_labels=len(labels),
  id2label={str(i): c for i, c in enumerate(labels)},
  label2id={c: str(i) for i, c in enumerate(labels)}
)

# Customizable training arguments
training_args = TrainingArguments(
  output_dir="./vit-base-birds",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=10,
  fp16=True,
  save_steps=10,
  eval_steps=10,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to="tensorboard",
  load_best_model_at_end=True,
)

# Stacks the inputs from a batch into a single tensor
def collate_fn(batch):
  return {
    "pixel_values" : torch.stack([x["pixel_values"] for x in batch]),
    "labels" : torch.tensor([x["label"] for x in batch])
  }

# Computes accuracy metric
metric = load("accuracy")
def compute_metrics(p):
  return metric.compute(predictions=np.argmax(p.predictions, axis=1),
                        references=p.label_ids)

# Create the trainer object
trainer = Trainer(
  model=model,
  args=training_args,
  data_collator=collate_fn,
  compute_metrics=compute_metrics,
  train_dataset=prepared_ds["train"],
  eval_dataset=prepared_ds["valid"],
  tokenizer=processor,
)

# Train the model 
train_results = trainer.train()

# Save the results
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# Finally, evaluate on the test set
metrics = trainer.evaluate(prepared_ds["test"])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

def get_attention_map(img, get_mask=False):
  # Pass the input through the model
  inputs  = processor(images=img, return_tensors="pt")
  print(torch.argmax(model(**inputs).logits, dim=1))
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

def plot_attention_map(original_img, att_map):
  fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
  ax1.set_title('Original')
  ax2.set_title('Attention Map Last Layer')
  _ = ax1.imshow(original_img)
  _ = ax2.imshow(att_map)
  plt.axis('off')
  plt.savefig("map.jpg")

# Plot attention map
img = Image.open(SPECTROGRAM_PATH + "/Barn Swallow/xc157331_039.jpg")
result = get_attention_map(img, True)
plot_attention_map(img, result)