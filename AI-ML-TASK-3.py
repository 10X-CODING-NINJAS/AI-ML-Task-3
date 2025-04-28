import pandas as pd
import numpy as np
import nltk
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

nlp = spacy.load('en_core_web_sm')

print("Loading dataset...")
df = pd.read_csv('IMDB Dataset.csv')
print("Dataset loaded successfully! 🗂️")
print(df.head())

stop_words = set(stopwords.words('english'))
extra_words = ['movie', 'film', 'one', 'make', 'like', 'get']
stop_words.update(extra_words)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

stemmer = PorterStemmer()
def stem_text(text):
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc if token.text not in stop_words]
    return ' '.join(lemmatized_words)

df_stemmed = df.copy()
df_lemmatized = df.copy()

print("Cleaning and removing stopwords... ⏳")
df_stemmed['review'] = df_stemmed['review'].apply(clean_text).apply(remove_stopwords)
df_lemmatized['review'] = df_lemmatized['review'].apply(clean_text).apply(remove_stopwords)

print("\nSample cleaned review (before stemming/lemmatizing):")
print(df_stemmed['review'].iloc[0])

print("\nApplying stemming...")
df_stemmed['review'] = df_stemmed['review'].apply(stem_text)

print("Applying lemmatization...")
df_lemmatized['review'] = df_lemmatized['review'].apply(lemmatize_text)

stemmed_vocab = set(' '.join(df_stemmed['review']).split())
lemmatized_vocab = set(' '.join(df_lemmatized['review']).split())

print(f"\n🧾 Vocabulary size after stemming: {len(stemmed_vocab)} words")
print(f"🧾 Vocabulary size after lemmatization: {len(lemmatized_vocab)} words")

def get_top_words(text_series, n=30):
    all_words = ' '.join(text_series).split()
    freq_dist = nltk.FreqDist(all_words)
    return freq_dist.most_common(n)

top30_stemmed = get_top_words(df_stemmed['review'])
top30_lemmatized = get_top_words(df_lemmatized['review'])

df_top30_stemmed = pd.DataFrame(top30_stemmed, columns=['word', 'count'])
df_top30_lemmatized = pd.DataFrame(top30_lemmatized, columns=['word', 'count'])

plt.figure(figsize=(12,6))
sns.barplot(x='count', y='word', data=df_top30_stemmed, palette='cubehelix')
plt.title('Top 30 Most Frequent Words (Stemmed)')
plt.xlabel('Word Count')
plt.ylabel('Words')
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(x='count', y='word', data=df_top30_lemmatized, palette='plasma')
plt.title('Top 30 Most Frequent Words (Lemmatized)')
plt.xlabel('Word Count')
plt.ylabel('Words')
plt.show()

print("\nGenerating wordcloud for stemmed text...")
wordcloud_stemmed = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df_stemmed['review']))

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_stemmed, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud - Stemmed Reviews')
plt.show()

print("Generating wordcloud for lemmatized text...")
wordcloud_lemmatized = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df_lemmatized['review']))

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_lemmatized, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud - Lemmatized Reviews')
plt.show()

print("\n🔎 Quick Analysis Summary:")
print("----------------------------------------------------")
print(f"✅ Vocabulary after stemming: {len(stemmed_vocab)} unique words.")
print(f"✅ Vocabulary after lemmatization: {len(lemmatized_vocab)} unique words.")
print("- Stemming chops words roughly (e.g., 'running' -> 'run'), sometimes loses meaning.")
print("- Lemmatization is smarter (e.g., 'running' -> 'run'), keeps more proper words.")
print("- Lemmatized text has slightly richer vocabulary and better readability.")
print("----------------------------------------------------")
