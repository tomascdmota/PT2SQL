from data_processing.data_processor import load_dataset, text_cleaning, tfidf_vectorization, filter_dataset_by_query
from models.model import train_random_forest, evaluate_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import spacy
import warnings
warnings.filterwarnings("ignore")
user_query = input("Enter your query: ")
# Load dataset
df = load_dataset('./dataset.csv')

# Remove rows with missing descriptions
df = df[df['Description'].notna()]

# Data preprocessing
df['Cleaned_Description'] = df['Description'].apply(text_cleaning)

# TF-IDF vectorization
tfidf_matrix, tfidf_vectorizer = tfidf_vectorization(df['Cleaned_Description'])

# Train the Random Forest model (Optional)
# train_data = df  # Use the entire dataset for training (optional)
# X_train = tfidf_vectorizer.transform(train_data['Cleaned_Description'])
# label_encoder = LabelEncoder()
# y_train = label_encoder.fit_transform(train_data['Title'])
# random_forest_model = train_random_forest(X_train, y_train)

# Process user query
filtered_df = filter_dataset_by_query(df, user_query)
recommended_books = filtered_df['Title'].tolist()

number_of_books = len(recommended_books)
print("Number of books:", number_of_books)
print("Recommended books:")
for book in recommended_books:
    print(book)
