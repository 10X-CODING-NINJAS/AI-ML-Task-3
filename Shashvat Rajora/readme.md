# 📚 NLP Preprocessing Case Study - IMDb Movie Reviews


## 📂 Dataset

- IMDb Dataset of 50K Movie Reviews (Binary Sentiment Classification - Positive/Negative)
- [Dataset Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 🛠️ Task Overview

- **Cleaned text**: Lowercasing, punctuation removal, digits removal, removing extra whitespaces.
- **Removed custom stopwords**:
  - Automatically generated using **TF-IDF** (low IDF terms).
  - Saved into `custom_stopwords.txt`.
- **Applied Stemming**:
  - Using **Porter Stemmer**.
  - Using **Snowball Stemmer**.
- **Applied Lemmatization**:
  - Using **WordNet Lemmatizer**.
- **Compared vocabulary size** after stemming and lemmatization.
- **Visualized top 30 frequent words** (Bar Plots).
- **Generated Word Clouds** for stemmed and lemmatized text.
- **Analysis** written comparing vocabulary richness and information retention.

---

## 🔥 Analysis

### ➡️ Stemming vs Lemmatization

| Aspect | Stemming | Lemmatization |
|:---|:---|:---|
| Method | Cuts word endings crudely | Uses dictionary and grammar rules |
| Speed | Faster | Slower |
| Accuracy | Less accurate | Highly accurate |
| Output | Sometimes non-words | Always valid dictionary words |

---

### ➡️ Types Used

- **Porter Stemmer** (Old but popular)
- **Snowball Stemmer** (Improved version of Porter)
- **WordNet Lemmatizer** (Grammar-based, POS tagging support)

---

### ➡️ Why Sometimes Stemming, Sometimes Lemmatization?

- **Stemming** is chosen when **speed** is more important (ex: search engines, quick filtering).
- **Lemmatization** is used when **interpretability and grammatical correctness** matter (ex: document understanding, summarization).

---

### ➡️ Why Custom Stopwords Were Used?

- Instead of guessing words manually, we used **TF-IDF scores** to **identify the most frequent and less informative words**.
- This ensures **domain-adaptive** stopword removal, improving model quality and cleaning efficiency.

---

## 📊 Visualizations

- Top 30 Words Bar Plots (Porter, Snowball, Lemmatization)
- Word Clouds (Porter Stemmed, Lemmatized)

---

## 📦 Files

| File | Description |
|:---|:---|
| `Task3_NLP_Processing.ipynb` | Main Colab Notebook |
| `custom_stopwords.txt` | Custom generated stopwords based on TF-IDF |

---

## 🚀 Conclusion

This project demonstrates the effectiveness of proper text preprocessing techniques and the trade-offs between stemming and lemmatization.  
It shows how automatic stopword selection (using TF-IDF) improves domain adaptation.

---

