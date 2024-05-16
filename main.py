from data_processing.data_processor import load_dataset, text_cleaning, tfidf_vectorization
from models.model import train_random_forest
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = load_dataset('./dataset.csv')

# Data preprocessing
df.fillna("", inplace=True)
df['Cleaned_Description'] = df['Description'].apply(text_cleaning)

# TF-IDF vectorization
tfidf_matrix, tfidf_vectorizer = tfidf_vectorization(df['Cleaned_Description'])

# Split dataset
train_size = int(0.4 * len(df))
val_size = int(0.02 * len(df))
test_size = len(df) - train_size - val_size

train_data = df[:train_size]
val_data = df[train_size:train_size + val_size]
test_data = df[train_size + val_size:]
X_train = tfidf_vectorizer.transform(train_data['Cleaned_Description'])
y_train = train_data['Category']
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Train the Random Forest model
random_forest_model = train_random_forest(X_train, y_train_encoded)
print("Training the Random Forest model...")

# Encode validation labels using the same label encoder used for training
label_encoder_val = LabelEncoder()
label_encoder_val.classes_ = label_encoder.classes_  # Ensure consistency with training labels
y_val_encoded = label_encoder_val.transform(val_data['Category'])

# Evaluate the model
X_val = tfidf_vectorizer.transform(val_data['Cleaned_Description'])
y_val_encoded = label_encoder.transform(val_data['Category'])
accuracy = evaluate_model(model, X_val, y_val_encoded)
print("Validation accuracy:", accuracy)