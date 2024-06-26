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
> **NOTE:** to ensure consistency, we recommend that you run all the steps below inside the project root directory, i.e., `<path-to>/csc413-project`.
2. Create a new Python virtual environment, and activate it.
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install all required dependencies.
```bash
pip install -r requirements.txt
```
> **NOTE:** We have provided a ready-to-use dataset with all of our preprocessing steps applied [here](https://drive.google.com/file/d/1Ekf4ZFG2Y-iPDbEymr3UNZuNHOblTwav/view?usp=drive_link). If you wish to use that, simply download the file and extract it into the project directory. You can then skip steps 4 and 5.

4. [Optional] Follow the instructions [here](https://www.kaggle.com/docs/api) to set up your environment to use Kaggle's API. You essentially just need to create a file called `~/.kaggle/kaggle.json` with the following format:
```
{"username":"your-username","key":"your-api-key"}
```
5. [Optional] Download the dataset, and apply all preprocessing steps (segmenting, resampling, converting to spectrograms, cropping, etc.). **This may take a while.**
```
python3 src/create_dataset.py
```
6. Run the ResNet50 based classifier. A script is provided to facilitate running on the GPU cluster provided by the teaching labs. This will train the model, evaluate it on a test set, and plot loss and validation curves. Simply run:
```
./run-cnn.sh
```
7. Similarly, we provide another script to run the ViT based classifier. In addition to the actions described above, this will also plot a sample attention map as shown in our paper.
```
./run-transformer.sh
```

> **NOTE:** We also provide a copy of our fine-tuned model weights [here](https://drive.google.com/file/d/1mRlfw0ClbyWbTl4PkQ1SAlJOqFOXA9vE/view?usp=sharing), for ViT. You can download this file, extract it in your working directory, and run `python3 src/transformer.py -a -e -p` to skip training and just evaluate and plot the attention map and loss/accuracy curves. 

## References

[Fine-Tune ViT for Image Classification with 🤗 Transformers](https://huggingface.co/blog/fine-tune-vit)

[Hugging Face - Create a dataset](https://huggingface.co/docs/datasets/en/create_dataset)

[Hugging Face - Process](https://huggingface.co/docs/datasets/en/process)

[Quickstart - Vision](https://huggingface.co/docs/datasets/en/quickstart#vision)

[Vision Transformer (ViT) : Visualize Attention Map](https://www.kaggle.com/code/piantic/vision-transformer-vit-visualize-attention-map)

[Data Augmentation Techniques for Audio Data in Python](https://towardsdatascience.com/data-augmentation-techniques-for-audio-data-in-python-15505483c63c)

[Audio data augmentation](https://www.kaggle.com/code/CVxTz/audio-data-augmentation/notebook)

[How do I split an audio file into multiple?](https://unix.stackexchange.com/a/283547)

[Guide to Audio Classification Using Deep Learning](https://www.analyticsvidhya.com/blog/2022/04/guide-to-audio-classification-using-deep-learning/)






