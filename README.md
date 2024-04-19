# Image Based Bird Song Classifcation

In this study, we explore the challenge of classifying bird species based on audio bird signals using computer vision image classification models — ResNet50 and Vision Transformer (ViT). We convert the audio files into image spectrograms that can be used as input into ResNet50 and ViT. The dataset considered is the [British Birdsong Dataset](https://www.kaggle.com/datasets/rtatman/british-birdsong-dataset), which features a robust collection of bird songs from 88 species of bird native to Britain.

This repository contains all of our source code to setup, train and run the model.

## Running the Code

The steps below are targeted for a Linux system. After following these instructions, you will be able to gather the source code and dataset, as well as train, evaluate, and visualize the model.

1. Clone the repository and change to the project directory.
   
```bash
git clone https://github.com/williamwang256/csc413-project
cd csc413-project
```
2. Create a new Python virtual environment, and activate it.
```bash
python3 -m venv venv
source venv
```
3. Install all required dependencies.
```bash
pip install -r requirements.txt
```
4. Set the following environment variable. This tells our scripts where all the data and models should live. Note that the is set for you if you choose to run one of the demo scripts we have included.
```bash
export CSC413_PROJECT_DIR="$PWD"  # store datasets and models in the current working directory
```
> **NOTE:** We have provided a ready-to-use dataset with all of our preprocessing steps applied here. If you wish to use that, simply download the file and extract it into the project directory. You can then skip steps 5 and 6.

5. [Optional] Follow the instructions [here](https://www.kaggle.com/docs/api) to set up your environment to use Kaggle's API. You essentially just need to create a file called `~/.kaggle/kaggle.json` with the following format:
```
{"username":"your-username","key":"your-api-key"}
```
6. [Optional] Download the dataset, and apply all proprocessing steps (segmenting, resampling, converting to spectrograms, cropping, etc.). **This may take a while.**
```
python3 src/create_dataset.py
```
7. Run the ResNet50 based classifier. A script is provided to facilitate running on the GPU cluster provided by the teaching labs. This will train the model, evaluate it on a test set, and plot loss and validation curves. Simply run:
```
./run-cnn.sh
```
8. Similarly, we provide another script to run the ViT based classifer. In addition to the actions described above, this will also plot a sample attention map as shown in our paper.
```
./run-transformer.sh
```

> We also provide a copy of our fine-tuned model weights here, for ViT. You can download this file, extract it in your working directory, and run `python3 transformers.py -a -e -p` to skip training and just evaluate and plot the attention map and loss/accuracy curves. 

## References

[Fine-Tune ViT for Image Classification with 🤗 Transformers](https://huggingface.co/blog/fine-tune-vit)

[Hugging Face - Create a dataset](https://huggingface.co/docs/datasets/en/create_dataset)

[Quickstart - Vision](https://huggingface.co/docs/datasets/en/quickstart#vision)

[Vision Transformer (ViT) : Visualize Attention Map](https://www.kaggle.com/code/piantic/vision-transformer-vit-visualize-attention-map)

[Data Augmentation Techniques for Audio Data in Python](https://towardsdatascience.com/data-augmentation-techniques-for-audio-data-in-python-15505483c63c)

[Audio data augmentation](https://www.kaggle.com/code/CVxTz/audio-data-augmentation/notebook)

[How do I split an audio file into multiple?](https://unix.stackexchange.com/a/283547)

[Guide to Audio Classification Using Deep Learning](https://www.analyticsvidhya.com/blog/2022/04/guide-to-audio-classification-using-deep-learning/)






