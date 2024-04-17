# https://huggingface.co/blog/fine-tune-vit
# https://huggingface.co/docs/datasets/en/create_datasethttps://huggingface.co/docs/datasets/en/create_dataset
# https://huggingface.co/docs/datasets/en/quickstart#vision
# https://www.kaggle.com/code/piantic/vision-transformer-vit-visualize-attention-map

import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from random import randint, uniform
import sys
import torch
from evaluate import load
from datasets import load_dataset, DatasetDict
from transformers import Trainer, TrainingArguments, ViTForImageClassification, ViTImageProcessor

from config import *

# We will fine-tune the following pre-trained model
MODEL = "google/vit-base-patch16-224-in21k"

# Metadata
df = pd.read_csv(METADATA_PATH, header=0)

# Performs time and frequency masking as a form of data augmentation on the spectrogram.
# Referenced from: https://www.kaggle.com/code/CVxTz/audio-data-augmentation/notebook
# NOTE: we chose not to apply this transformation to the final model, but we leave the code here for completeness.
def spec_augment(original_melspec, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
  augmented_melspec = original_melspec.clone()
  # print(augmented_melspec.shape)
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

# Load the dataset, and split into train, test, and validation sets
# Use the split: 60% train, 20% validation, 20% test
def load_ds():
  full = load_dataset("imagefolder", data_dir=SPECTROGRAM_PATH, split="train")
  train_testvalid = full.train_test_split(test_size=0.4)
  test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
  ds = DatasetDict({
    "train": train_testvalid["train"],
    "test": test_valid["test"],
    "valid": test_valid["train"]})
  return ds

# A transformation which applies the above processor and some data
# augmentation steps (time shifting)
def transform(example_batch):
  inputs = processor([x for x in example_batch["image"]], return_tensors="pt")
  inputs["pixel_values"] = torch.roll(inputs["pixel_values"], randint(1, 10000), dims=3)
  # inputs["pixel_values"] = spec_augment(inputs["pixel_values"])
  inputs["label"] = example_batch["label"]
  return inputs

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
  plt.savefig(os.path.join(PLOTS_DIR, "attn_map.jpg"))


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
  
  ds = load_ds()
  prepared_ds = ds.with_transform(transform)

  if ("-t") in sys.argv:
    print("Training new model...")
    labels = ds["train"].features["label"].names
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
    attn_map = get_attention_map(model, processor, img, True)
    plot_attention_map(img, attn_map)

  if ("-p" in sys.argv):
    if not os.path.isdir(MODEL_SAVE_DIR):
      print("No saved model found.")
      exit(1)
    print("Plotting accuacy and loss curves for saved model: ", MODEL_SAVE_DIR)
    plot_curves()