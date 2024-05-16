import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_dataset(file_path):
    # Load dataset from CSV file
    df = pd.read_csv(file_path)
    return df

def text_cleaning(text):
    # Perform text cleaning: lowercase, remove punctuation, and remove stopwords
    text = text.lower()
    text = text.replace(",", "")  # Remove commas
    text = text.replace(".", "")  # Remove periods
    # Additional cleaning steps (e.g., removing special characters, digits) can be added here
    return text

def tokenize(text):
    # Tokenize the cleaned text by splitting on whitespace
    return text.split()

def tfidf_vectorization(descriptions):
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize)
    # Fit TF-IDF vectorizer on the descriptions and transform them into TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)
    return tfidf_matrix, tfidf_vectorizer
