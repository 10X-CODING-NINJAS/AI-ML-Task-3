# IMDB Sentiment Analysis - AI/ML Task 3

## Overview

This project focuses on sentiment analysis of movie reviews from the **IMDB** dataset using machine learning techniques. The goal is to build a model that predicts whether a movie review is **positive** or **negative** based on the text content.

## Technologies and Libraries

This project is built using Python and the following libraries:

- **Python 3.x**
- **NumPy** - For numerical operations.
- **Pandas** - For data handling and manipulation.
- **scikit-learn** - For implementing machine learning algorithms.
- **NLTK** (Natural Language Toolkit) - For natural language processing tasks.
- **TensorFlow** or **Keras** - For deep learning models.
- **Matplotlib** - For data visualization.

## Dataset

The dataset used in this project is the **IMDB movie reviews** dataset. It contains 50,000 labeled reviews—25,000 for training and 25,000 for testing. The labels are binary: `1` for positive and `0` for negative reviews.

- **Source**: [Stanford IMDB Sentiment Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

### 1. Clone the Repository

To begin, clone the repository to your local machine:

```bash
git clone https://github.com/krishnakeshab-banik/AI-ML-Task-3.git
cd AI-ML-Task-3
```

### 2. Install Dependencies

Install the required libraries by running:

```bash
pip install -r requirements.txt
```

Alternatively, you can install the dependencies manually:

```bash
pip install numpy pandas scikit-learn nltk tensorflow
```

### 3. Dataset Preparation

The IMDB dataset should be downloaded separately from the [official source](https://ai.stanford.edu/~amaas/data/sentiment/). After downloading, place the dataset in the appropriate directory as specified in the code.

## Workflow

### 1. Data Preprocessing

- **Tokenization**: Split text data into individual words (tokens).
- **Vectorization**: Convert text into numerical data using techniques like **TF-IDF** or **Bag of Words**.
- **Cleaning**: Remove irrelevant characters, stop words, and special symbols from the text data.

### 2. Model Building

We build and train a machine learning model using either traditional algorithms like **Logistic Regression** or **Naive Bayes**, or by employing deep learning models like **LSTM** (Long Short-Term Memory) or **GRU** (Gated Recurrent Units).

### 3. Model Evaluation

We evaluate the performance of the model using metrics such as:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

### 4. Sentiment Prediction

Once trained, the model is used to predict the sentiment of new, unseen movie reviews.

## How to Run

### Training the Model

To train the sentiment analysis model on the IMDB dataset, run the following script:

```bash
python train_model.py
```

This script loads the dataset, processes the text data, trains the model, and saves it for future use.

### Predicting Sentiment

After the model is trained, use the following script to predict the sentiment of a new movie review:

```bash
python predict_sentiment.py "The movie was amazing! I loved it."
```

The model will return either **positive** or **negative** based on the review's sentiment.
