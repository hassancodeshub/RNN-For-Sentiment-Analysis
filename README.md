# IMDB Movie Review Sentiment Analysis with RNN

A learning project exploring sentiment classification on movie reviews from the IMDB dataset using text preprocessing, TF-IDF vectorization, and a Recurrent Neural Network (RNN) built with PyTorch.

## Project Overview

This notebook walks through a basic NLP workflow for classifying movie reviews as **positive** or **negative**.

### Workflow

1. Load the IMDB dataset from `IMDB Dataset.csv`.
2. Inspect the data and remove duplicate reviews.
3. Preprocess review text:
   - Convert text to lowercase
   - Remove URLs, punctuation, and HTML tags
   - Remove stopwords
   - Apply stemming
4. Encode sentiment labels numerically.
5. Convert review text into TF-IDF features.
6. Split the dataset into training and testing sets.
7. Create PyTorch datasets and data loaders.
8. Define and train an RNN model.
9. Evaluate the model using classification accuracy.

## Technologies Used

- Python
- Pandas
- NLTK
- scikit-learn
- PyTorch
- Jupyter Notebook

## Requirements

Install the main dependencies:

```bash
pip install pandas nltk scikit-learn torch jupyter
```

Download the NLTK resources used by the notebook in Python:

```python
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

## Dataset

Place `IMDB Dataset.csv` in the project directory before running the notebook. The CSV is expected to contain `review` and `sentiment` columns.

## How to Run

1. Clone or download this repository.
2. Place the dataset file in the project directory.
3. Install the dependencies listed above.
4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `rnn.ipynb` and run the cells in order.

## Learning Goals

- Practice text cleaning and preprocessing for NLP
- Explore label encoding and TF-IDF feature extraction
- Learn how to prepare data for PyTorch
- Build a simple RNN classifier
- Understand a basic model training and evaluation workflow

## Note

This repository is intended for learning and experimentation with natural language processing and neural networks.
