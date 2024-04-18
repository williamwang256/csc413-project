import os

try:
  BASE = os.environ["CSC413_PROJECT_DIR"]
  SONGS_PATH = BASE + "/data/songs/songs/"
  METADATA_PATH = BASE + "/data/birdsong_metadata.csv"
  AUDIO_PATH = BASE + "/data/audio/"
  SPECTROGRAM_PATH = BASE + "/data/spectrograms/"
  PLOTS_DIR = BASE + "/plots/"
except KeyError:
  print("CSC413_PROJECT_DIR not detected. "
        "Please make sure to set the CSC413_PROJECT_DIR environment variable "
        "to your current working directory before running.")
  exit()