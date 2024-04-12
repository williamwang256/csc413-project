# https://huggingface.co/blog/fine-tune-vit
# https://huggingface.co/docs/datasets/en/create_datasethttps://huggingface.co/docs/datasets/en/create_dataset
# https://huggingface.co/docs/datasets/en/quickstart#vision

import torch
import numpy as np
from datasets import load_metric, load_dataset
from transformers import Trainer, TrainingArguments, ViTForImageClassification, ViTImageProcessor

from config import *


def transform(example_batch):
  inputs = processor([x for x in example_batch["image"]], return_tensors="pt")
  inputs["label"] = example_batch["label"]
  return inputs


def collate_fn(batch):
  return {
    "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
    "labels": torch.tensor([x["label"] for x in batch])
  }


def compute_metrics(p):
  return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


if __name__ == "__main__":

  ds = load_dataset("imagefolder", data_dir=SORTED_PATH, split="train")
  ds = ds.train_test_split(test_size=0.2)

  model_name_or_path = "google/vit-base-patch16-224-in21k"
  processor = ViTImageProcessor.from_pretrained(model_name_or_path)

  prepared_ds = ds.with_transform(transform)
  metric = load_metric("accuracy")
  labels = ds["train"].features["label"].names

  model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
  )

  training_args = TrainingArguments(
    output_dir="./vit-base-birds",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=10,
    fp16=True,
    save_steps=100,
    eval_steps=100,
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
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["test"],
    tokenizer=processor,
  )

  train_results = trainer.train()
  trainer.save_model()
  trainer.log_metrics("train", train_results.metrics)
  trainer.save_metrics("train", train_results.metrics)
  trainer.save_state()

  metrics = trainer.evaluate(prepared_ds["test"])
  trainer.log_metrics("eval", metrics)
  trainer.save_metrics("eval", metrics)