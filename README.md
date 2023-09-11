# BERT Sentiment Analysis with Hugging Face Transformers

## BERT Sentiment Analysis

This repository contains a Sentiment Analysis project using the BERT model from Hugging Face Transformers. We perform sentiment classification on text data using transfer learning with a random sampler and the AdamW optimizer.
Table of Contents
- Introduction
-   Requirements
-   Installation
-   Usage
-   Results

## Introduction

Sentiment Analysis is the process of determining the sentiment or emotional tone of a piece of text, whether it's positive, negative, or neutral. BERT (Bidirectional Encoder Representations from Transformers) is a state-of-the-art natural language processing model that has shown excellent performance on various NLP tasks, including sentiment analysis.

In this project, we fine-tune a pre-trained BERT model on a sentiment analysis dataset using transfer learning. We also utilize a random sampler to handle the imbalanced nature of the dataset and the AdamW optimizer for training stability.
Requirements

Before running the code in this repository, make sure you have the following requirements installed:
```bash
    Python 3.x
    PyTorch
    Hugging Face Transformers
    matplotlib
    seaborn
    NumPy
    Pandas
```
You can install these packages using pip:

```bash
pip install torch transformers numpy pandas matplotlib seaborn scikit-learn
```
## Installation

To get started with this project, follow these steps:

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/bert-sentiment-analysis.git
cd bert-sentiment-analysis
```
Install the required packages as mentioned in the Requirements section.

    Download the sentiment analysis dataset from https://www.kaggle.com/datasets/ashkhagan/smile-twitter-emotion-dataset

## Usage

To train and evaluate the sentiment analysis model with BERT and Save the model



    You can also use the saved and trained model to perform sentiment analysis on new text data 

## Results

The model's performance and evaluation results after training the model for 10 epochs the Total loss and Test Accuracy :
```bash
Training Loss: 0.1455424027163771
Test Score:0.8505631660263822
```

## Contributing

We welcome contributions to this project. If you have any suggestions or want to report issues, please open an issue on GitHub. If you'd like to contribute code, please fork the repository, create a feature branch, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

