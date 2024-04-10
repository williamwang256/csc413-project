import csv
import os
import random
import re
import subprocess

import matplotlib.pyplot as plt
import librosa.display
import numpy as np

SEGMENT_TIME = 4
SAMPLE_RATE = 44100
NUM_SAMPLES = 179712

BASE = "/h/u6/c9/01/wangwi18/winter24/csc413/project/"
DATA_PATH = BASE + "/data/"
OUTPUT_PATH = BASE + "/output/"
LABELS_PATH = BASE + "/metadata/labels.csv"

SPECTROGRAM_PATH = BASE + "/spectrograms/"
SPEC_AUGMENT_PATH = BASE + "/augmentations/spec_augment/"
TIME_SHIFT_PATH = BASE + "/augmentations/time_shift/"

fid_to_label = {}
english_names = []

def initialize():
  with open(LABELS_PATH, newline="") as f:
    r = csv.reader(f, delimiter=",")
    for i, row in enumerate(r):
      fids = row[1].split(",")
      for fid in fids:
        fid_to_label[fid] = i
      english_names.append(row[0])

# Runs ffmpeg to separate audio clips longer than SEGMENT_TIME into segments of length SEGMENT_TIME
def segment_clips():
  pattern = "xc([0-9]*)\.flac"
  for file in os.listdir(DATA_PATH):
    m = re.search(pattern, file)
    if m is not None:
      filename, fid = m.group(0), m.group(1)
      command = (f"ffmpeg -i {DATA_PATH}/{filename} -f segment -segment_time {SEGMENT_TIME} {OUTPUT_PATH}/{fid}_%03d_{fid_to_label[fid]}.wav")
      process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      process.wait()


# Pads the given signal y up to a length of NUM_SAMPLES
def pad_signal(y):
  pad_amount = NUM_SAMPLES - y.shape[0]
  return np.pad(y, (0, pad_amount), "constant", constant_values=(0, 0))


# Performs time and frequency masking as a form of data augmentation on the spectrogram.
# Referenced from: https://www.kaggle.com/code/CVxTz/audio-data-augmentation/notebook
def spec_augment(original_melspec, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
    augmented_melspec = original_melspec.copy()
    all_frames_num, all_freqs_num = augmented_melspec.shape

    # Frequency masking
    freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
    num_freqs_to_mask = int(freq_percentage * all_freqs_num)
    f0 = int(np.random.uniform(low = 0.0, high = (all_freqs_num - num_freqs_to_mask)))
    augmented_melspec[:, f0:(f0 + num_freqs_to_mask)] = 0

    # Time masking
    time_percentage = random.uniform(0.0, time_masking_max_percentage)
    num_frames_to_mask = int(time_percentage * all_frames_num)
    t0 = int(np.random.uniform(low = 0.0, high = (all_frames_num - num_frames_to_mask)))
    augmented_melspec[t0:(t0 + num_frames_to_mask), :] = 0
    
    return augmented_melspec


# Shifts the audio in time using np.roll
def time_shift_augment(original_melspec, shift_amt=3000):
  return np.roll(original_melspec, shift_amt)


# Plot the given spectrogram and save under the provided filename and path
def plot_spectrogram(S, sr, filename, path):
  fig, ax = plt.subplots()
  librosa.display.specshow(S, sr=sr, fmax=8000, ax=ax)
  fig.tight_layout()
  fig.savefig(f"{path}/{filename}.png")
  plt.axis("off")
  plt.close()


def generate_spectrograms():
  for file in os.listdir(OUTPUT_PATH):
    y, sr = librosa.load(f"{OUTPUT_PATH}/{file}", sr=SAMPLE_RATE, mono=True)
    y = pad_signal(y)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=500, fmax=15000, n_fft=2048)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB_spec_augment = spec_augment(S_dB)
    S_dB_time_shift = time_shift_augment(S_dB)

    filename = file.strip(".wav")

    plot_spectrogram(S=S_dB, sr=sr, filename=filename, path=SPECTROGRAM_PATH)
    plot_spectrogram(S=S_dB_spec_augment, sr=sr, filename=filename, path=SPEC_AUGMENT_PATH)
    plot_spectrogram(S=S_dB_time_shift, sr=sr, filename=filename, path=TIME_SHIFT_PATH)

    
if __name__ == "__main__":
  initialize()
  # segment_clips()
  generate_spectrograms()