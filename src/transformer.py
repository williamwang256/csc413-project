# https://huggingface.co/blog/fine-tune-vit
# https://huggingface.co/docs/datasets/en/create_datasethttps://huggingface.co/docs/datasets/en/create_dataset
# https://huggingface.co/docs/datasets/en/quickstart#vision

import torch
import numpy as np
from random import randint
from datasets import load_metric, load_dataset, DatasetDict
from transformers import Trainer, TrainingArguments, ViTForImageClassification, ViTImageProcessor

from config import *


# We will fine-tune the following pre-trained model
model_name_or_path = "google/vit-base-patch16-224-in21k"

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
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

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
  model_name_or_path,
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
metric = load_metric("accuracy")
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