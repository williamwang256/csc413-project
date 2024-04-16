# https://unix.stackexchange.com/a/283547
# https://www.analyticsvidhya.com/blog/2022/04/guide-to-audio-classification-using-deep-learning/
# https://stackoverflow.com/a/60309843

import os
import re
import subprocess

import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from kaggle.api.kaggle_api_extended import KaggleApi

from config import *

SEGMENT_TIME = 4
SAMPLE_RATE = 44100
NUM_SAMPLES = 179712


def download_dataset():
  print("Downloading dataset...")
  api = KaggleApi()
  api.authenticate()
  api.dataset_download_files("rtatman/british-birdsong-dataset", path=BASE, unzip=True)


def create_folders():
  print("Creating folders...")
  os.makedirs(AUDIO_PATH, exist_ok=True)
  os.makedirs(SPECTROGRAM_PATH, exist_ok=True)


def segment_clips():
  print("Segmenting clips...")
  pattern = "(xc[0-9]*)\.flac"
  for file in os.listdir(SONGS_PATH):
    m = re.search(pattern, file)
    if m is not None:
      filename, fid = m.group(0), m.group(1)
      command = f"ffmpeg -i {os.path.join(SONGS_PATH, filename)} -f segment -segment_time {SEGMENT_TIME} {os.path.join(AUDIO_PATH, fid + '_%03d.wav')}"
      process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      process.wait()


def get_english_name(fid):
  return df.loc[df["file_id"] == fid]["english_cname"].values[0]


def plot_spectrogram(S, sr, fid, segment):
  bird_name = get_english_name(int(fid.strip("xc")))
  path = os.path.join(SPECTROGRAM_PATH, bird_name)
  filename = os.path.join(path, f"{fid}_{segment}.jpg")
  os.makedirs(path, exist_ok=True)
  fig, ax = plt.subplots()
  librosa.display.specshow(S, sr=sr, fmin=500, fmax=15000, ax=ax)
  fig.tight_layout()
  fig.savefig(filename)
  plt.axis("off")
  plt.close()
  command = f"convert \"{filename}\" -crop 610x450+15+15 \"{filename}\""
  process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  process.wait()


def generate_spectrograms():
  print("Generating spectrograms...")
  for file in os.listdir(AUDIO_PATH):
    y, sr = librosa.load(f"{AUDIO_PATH}/{file}", sr=SAMPLE_RATE, mono=True)
    pad_amount = NUM_SAMPLES - y.shape[0]
    y = np.pad(y, (0, pad_amount), "constant", constant_values=(0, 0))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=500, fmax=15000, n_fft=2048)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fid, segment = file.strip(".wav").split("_")
    plot_spectrogram(S=S_dB, sr=sr, fid=fid, segment=segment)

    
if __name__ == "__main__":
  download_dataset()
  create_folders()
  global df
  df = pd.read_csv(METADATA_PATH, header=0)
  segment_clips()
  generate_spectrograms()