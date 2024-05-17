import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import logging
logging.basicConfig(level=logging.INFO)

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def load_dataset(file_path):
    # Load dataset from CSV file
    df = pd.read_csv(file_path)
    # Drop rows where the 'Description' column is missing
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

def extract_query_entities(query):
    doc = nlp(query)
    entities = []
    author_entities = []
    for ent in doc.ents:
        entities.append(ent.text.lower())
    # Extract additional keywords based on query structure
    for token in doc:
        if token.pos_ == 'PROPN' and token.dep_ == 'compound':
            entities.append(token.text.lower())
        elif token.pos_ == 'NUM' and token.dep_ == 'nummod':
            entities.append(token.text.lower())
        # Check for phrases indicating authorship
        if token.text.lower() in ["by", "from", "written", "authored", "created", "by:"]:
            for child in token.children:
                if child.ent_type_ == "PERSON":
                    author_entities.append(child.text.lower())
    # Combine the entities and author_entities lists, removing duplicates
    entities.extend([entity for entity in author_entities if entity not in entities])
    return entities

def filter_dataset_by_query(df, query):
    # Extract entities from the user query
    query_entities = extract_query_entities(query)
    print(query_entities)
    
    # Determine if the query indicates a search for authors
    search_for_author = any(entity.lower() in ['by', 'from', 'written by', 'authored by', 'by author'] for entity in query_entities)
    search_for_date = any(entity.lower() in ['from', 'since'] for entity in query_entities)
    
    # Filter the dataset based on the query entities
    filtered_rows = []  # Initialize an empty list to store filtered rows
    
    if search_for_author:
        for index, row in df.iterrows():
            # Check if the author's name matches any of the query entities
            match_found = any(entity.lower() in str(row['Authors']).lower() for entity in query_entities)
            if match_found:
                filtered_rows.append(row)
    else:
        for index, row in df.iterrows():
            # Check if any of the query entities are present in the title or description
            match_found = any(entity.lower() in str(row['Title']).lower() or
                              entity.lower() in str(row['Description']).lower() for entity in query_entities)
            if match_found:
                filtered_rows.append(row)
    
    # Create a DataFrame from the filtered rows
    filtered_df = pd.DataFrame(filtered_rows)
    
    return filtered_df
