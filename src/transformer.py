# https://huggingface.co/blog/fine-tune-vit
# https://huggingface.co/docs/datasets/en/create_dataset
# https://huggingface.co/docs/datasets/en/quickstart#vision
# https://www.kaggle.com/code/piantic/vision-transformer-vit-visualize-attention-map
# https://towardsdatascience.com/data-augmentation-techniques-for-audio-data-in-python-15505483c63c

import getopt
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
processor = ViTImageProcessor.from_pretrained(MODEL)

df = pd.read_csv(METADATA_PATH, header=0)

# Perform random time-shift augmentation on the given spectrogram 
def time_shift_augment(original_melspec):
  return torch.roll(original_melspec, randint(1, 10000), dims=2)

# Performs time and frequency masking as a form of data augmentation on the spectrogram.
# Referenced from: https://www.kaggle.com/code/CVxTz/audio-data-augmentation/notebook
# NOTE: we chose not to apply this transformation to the final model, but we leave the code here for completeness.
def spec_augment(original_melspec, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
  augmented_melspec = original_melspec.clone()
  _, _, all_frames_num, all_freqs_num = augmented_melspec.shape

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

# Tranform without augmentation (for validation and test sets)
def transform(example_batch):
  inputs = processor([x for x in example_batch["image"]], return_tensors="pt")
  inputs["label"] = example_batch["label"]
  return inputs

# Transform with time shift augmentation (for training set)
def transform_augment(example_batch):
  inputs = processor([x for x in example_batch["image"]], return_tensors="pt")
  inputs["pixel_values"] = time_shift_augment(inputs["pixel_values"])
  # inputs["pixel_values"] = spec_augment(inputs["pixel_values"])
  inputs["label"] = example_batch["label"]
  return inputs

# Load the dataset, and split into train, test, and validation sets
# Use the split: 60% train, 20% validation, 20% test
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

# Gets the trainer object
def get_trainer(model, processor, ds, save_dir):
  training_args = TrainingArguments(
    output_dir=save_dir,
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=10,
    fp16=torch.cuda.is_available(),  # fp16 only supported on GPU
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    # logging_strategy="no",
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
def plot_curves(save_dir):
  with open(os.path.join(save_dir, "trainer_state.json"), "r") as f:
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

def usage():
  print("Usage: python3 transformer.py [-t] [-a] [-e] [-p] [-h]")
  print("[-t <model_path>] Trains the model and saves to <model_path>\n"
        "[-a <model_path>] Outputs sample attention map for the saved model\n"
        "[-e <model_path>] Evaluates the saved model on the test set\n"
        "[-p <model_path>] Plots loss and accuracy curves for the saved model")

if __name__ == "__main__":
  try:
    opts, args = getopt.getopt(sys.argv[1:], "a:e:t:p:hl")
  except getopt.GetoptError as err:
    usage()
    exit(1)

  if len(opts) == 0:
    usage()
    exit()

  for o, a in opts:
    if o in ("-h"):
      usage()
      exit()

    elif o in ("-t"):
      print("Training new model...")
      save_dir = a
      ds = load_ds()
      labels = ds["train"].features["label"].names
      model = ViTForImageClassification.from_pretrained(
        MODEL,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
      )
      trainer = get_trainer(model, processor, ds, save_dir)
      train_results = trainer.train()
      trainer.save_model()
      trainer.log_metrics("train", train_results.metrics)
      trainer.save_metrics("train", train_results.metrics)
      trainer.save_state()

    elif o in ("-e"):
      save_dir = a
      if not os.path.isdir(save_dir):
        print("No saved model found.")
        exit()
      print("Evaluating on saved model: ", save_dir)
      ds = load_ds()
      model = ViTForImageClassification.from_pretrained(save_dir)
      trainer = get_trainer(model, processor, ds, save_dir)
      metrics = trainer.evaluate(ds["test"])
      trainer.log_metrics("eval", metrics)
      trainer.save_metrics("eval", metrics)

    elif o in ("-a"):
      save_dir = a
      if not os.path.isdir(save_dir):
        print("No saved model found.")
        exit(0)
      print("Generating attention map on saved model: ", save_dir)
      model = ViTForImageClassification.from_pretrained(save_dir)
      img = Image.open(SPECTROGRAM_PATH + "/Eurasian Wren/xc71024_021.jpg")
      attn_map = get_attention_map(model, processor, img, True)
      plot_attention_map(img, attn_map)

    elif o in ("-p"):
      save_dir = a
      if not os.path.isdir(save_dir):
        print("No saved model found.")
        exit()
      print("Plotting accuacy and loss curves for saved model: ", save_dir)
      plot_curves(save_dir)

    else:
      assert False, "Unknown option!"