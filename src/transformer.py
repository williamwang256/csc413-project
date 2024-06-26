# References:
# [1] https://huggingface.co/blog/fine-tune-vit
# [2] https://huggingface.co/docs/datasets/en/create_dataset
# [3] https://huggingface.co/docs/datasets/en/quickstart#vision
# [4] https://towardsdatascience.com/data-augmentation-techniques-for-audio-data-in-python-15505483c63c
# [5] https://www.kaggle.com/code/CVxTz/audio-data-augmentation/notebook
# [6] https://www.kaggle.com/code/piantic/vision-transformer-vit-visualize-attention-map
# [7] https://huggingface.co/docs/datasets/en/process

import json
import os
from random import randint, uniform
import sys

import cv2
from datasets import load_dataset, DatasetDict
from evaluate import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import Trainer, TrainingArguments, ViTForImageClassification, ViTImageProcessor

from config import *

# We will fine-tune the following pre-trained model
MODEL = "google/vit-base-patch16-224-in21k"

# Where to save the model
MODEL_SAVE_DIR = os.path.join(BASE, "vit-birds")

# Metadata
df = pd.read_csv(METADATA_PATH, header=0)

# Performs time and frequency masking as a form of data augmentation on the
# spectrogram. Reference: [4] and [5] NOTE: we chose not to apply this
# transformation to the final model, but we leave the code here for completeness.
def spec_augment(original_melspec, freq_masking_max_percentage=0.15,
                 time_masking_max_percentage=0.3):
  augmented_melspec = original_melspec.clone()
  _, _, all_frames_num, all_freqs_num = augmented_melspec.shape

  # Frequency masking
  freq_percentage = uniform(0.0, freq_masking_max_percentage)
  num_freqs_to_mask = int(freq_percentage * all_freqs_num)
  f0 = int(np.random.uniform(low=0.0, high=(all_freqs_num - num_freqs_to_mask)))
  augmented_melspec[:, :, f0:(f0 + num_freqs_to_mask)] = 0

  # Time masking
  time_percentage = uniform(0.0, time_masking_max_percentage)
  num_frames_to_mask = int(time_percentage * all_frames_num)
  t0 = int(np.random.uniform(low=0.0, high=(all_frames_num - num_frames_to_mask)))
  augmented_melspec[:, t0:(t0 + num_frames_to_mask), :] = 0

  return augmented_melspec

# Tranform without augmentation (for validation and test sets). Reference: [1]
def transform(example_batch):
  inputs = processor([x for x in example_batch["image"]], return_tensors="pt")
  inputs["label"] = example_batch["label"]
  return inputs

# Transform with time shift augmentation (for training set). Reference: [1], [4]
# and [5]
def transform_augment(example_batch):
  inputs = processor([x for x in example_batch["image"]], return_tensors="pt")
  inputs["pixel_values"] = \
    torch.roll(inputs["pixel_values"], randint(1, 10000), dims=3) # time shift
  inputs["label"] = example_batch["label"]
  return inputs

# Load the dataset and split into train/test/validation sets. Use the split:
# 60% train, 20% validation, 20% test. Reference: [2], [3] and [7]
def load_ds():
  # Load the full dataset, making sure to shuffle it (deterministically, by setting
  # a seed) to ensure classes are equally represented when we perform the
  # train/valid/test split.
  full = load_dataset(
    "imagefolder", data_dir=SPECTROGRAM_PATH, split="train"
  ).shuffle(seed=26).flatten_indices()

  # Separate out the test set. Don't shuffle again to ensure no training or
  # validation data is ever part of the test set.
  split1 = full.train_test_split(test_size=0.2, shuffle=False)

  # Separate out the validation set. This time we do shuffle since the saved model
  # won't be evaluated against this set of data again.
  split2 = split1["train"].train_test_split(test_size=0.25, shuffle=True)

  return DatasetDict({
    "train": split2["train"].with_transform(transform_augment),
    "test":  split1["test"].with_transform(transform),
    "valid": split2["test"].with_transform(transform)})

# Stacks the inputs from a batch into a single tensor. Reference: [1]
def collate_fn(batch):
  return {
    "pixel_values" : torch.stack([x["pixel_values"] for x in batch]),
    "labels" : torch.tensor([x["label"] for x in batch])
  }

# Computes accuracy metric. Reference: [1]
metric = load("accuracy")
def compute_metrics(p):
  return metric.compute(predictions=np.argmax(p.predictions, axis=1),
                        references=p.label_ids)

# Gets the trainer object. Reference: [1]
def get_trainer(model, processor, ds):
  training_args = TrainingArguments(
    output_dir=MODEL_SAVE_DIR,
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=10,
    fp16=torch.cuda.is_available(),  # fp16 only supported on GPU
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
  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=ds["train"],
    eval_dataset=ds["valid"],
    tokenizer=processor,
  )
  return trainer

# Uses the model to make a prediction for the given image
def predict(img, model, processor):
  inputs = processor(images=img, return_tensors="pt")
  idx = int(torch.argmax(model(**inputs).logits, dim=1)[0])
  return sorted(list(set(df["english_cname"])))[idx]

# Obtains the attention map for the given image. Reference: [6]
def get_attention_map(model, processor, img):
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
  return cv2.resize(mask / mask.max(), img.size)

# Plots the attention map alongside the original image. Reference: [6]
def plot_attention_map(original_img, att_map):
  _, (ax1, ax2) = plt.subplots(ncols=2)
  ax1.set_title("Original")
  ax2.set_title("Attention Map Last Layer")
  _ = ax1.imshow(original_img)
  _ = ax2.imshow(att_map)
  ax1.axis("off")
  ax2.axis("off")
  plt.tight_layout()
  plt.savefig(os.path.join(PLOTS_DIR, "attn_map.png"))

# Plots the training loss and validation accuracy curves
def plot_curves():
  with open(os.path.join(MODEL_SAVE_DIR, "trainer_state.json"), "r") as f:
    d = json.load(f)
    eval_checkpoints = []
    train_checkpoints = []
    acc = []
    loss = []
    for entry in d["log_history"]:
      if "eval_accuracy" in entry:
        eval_checkpoints.append(entry["epoch"])
        acc.append(entry["eval_accuracy"])
      if ("loss") in entry:
        train_checkpoints.append(entry["epoch"])
        loss.append(entry["loss"])
    plt.figure()
    plt.plot(eval_checkpoints, acc)
    plt.title("Validation Accuracy over Iterations")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(PLOTS_DIR, "transformers_acc.png"))
    plt.figure()
    plt.plot(train_checkpoints, loss)
    plt.title("Training Loss over Iterations")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.savefig("plots/transformers_loss.png")

if __name__ == "__main__":
  
  if len(sys.argv) < 2 or len(sys.argv) > 5:
    print("Usage: python3 transformer.py [-t] [-a] [-e] [-p] [-h]")
    exit(1)
  
  if ("-h" in sys.argv):
    print("[-t] Trains the model\n"
          "[-a] Outputs sample attention map\n"
          "[-e] Evaluates on the test set\n"
          "[-p] Plots loss and accuracy curves")
    exit(1)

  processor = ViTImageProcessor.from_pretrained(MODEL)
  prepared_ds =  load_ds()

  if ("-t") in sys.argv:
    print("Training new model...")
    labels = prepared_ds["train"].features["label"].names
    model = ViTForImageClassification.from_pretrained(
      MODEL,
      num_labels=len(labels),
      id2label={str(i): c for i, c in enumerate(labels)},
      label2id={c: str(i) for i, c in enumerate(labels)}
    )
    trainer = get_trainer(model, processor, prepared_ds)
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

  if ("-e") in sys.argv:
    if not os.path.isdir(MODEL_SAVE_DIR):
      print("No saved model found.")
      exit(1)
    print("Evaluating on saved model: ", MODEL_SAVE_DIR)
    model = ViTForImageClassification.from_pretrained(MODEL_SAVE_DIR)
    trainer = get_trainer(model, processor, prepared_ds)
    metrics = trainer.evaluate(prepared_ds["test"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

  if ("-a" in sys.argv):
    if not os.path.isdir(MODEL_SAVE_DIR):
      print("No saved model found.")
      exit(1)
    print("Generating attention map on saved model: ", MODEL_SAVE_DIR)
    model = ViTForImageClassification.from_pretrained(MODEL_SAVE_DIR)
    img = Image.open(SPECTROGRAM_PATH + "/Eurasian Wren/xc71024_021.jpg")
    attn_map = get_attention_map(model, processor, img)
    plot_attention_map(img, attn_map)

  if ("-p" in sys.argv):
    if not os.path.isdir(MODEL_SAVE_DIR):
      print("No saved model found.")
      exit(1)
    print("Plotting accuacy and loss curves for saved model: ", MODEL_SAVE_DIR)
    plot_curves()