import pandas as pd
import os
import torch
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer, BartConfig
from googletrans import Translator
from langdetect import detect
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Initialize the Translator
translator = Translator()

# Initialize the NL2SQL pipeline
#model_name = "SwastikM/bart-large-nl2sql"
pre_trained_model_dir = './trained_model'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Check if GPU is available and move model to GPU if it is
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Model is using device: {model.device}")

# Define a function to translate a question from Portuguese to English
def translate_question(question):
    if detect(question) == 'en':
        return question  # Return the question unchanged if it's already in English
    else:
        translated_text = translator.translate(question, src='pt', dest='en')
        return translated_text.text

# Define a function to generate SQL query from English question using the pipeline
def generate_sql_query(question):
    input_text = question
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)  # Move input tensors to GPU
    generated_ids = model.generate(input_ids)
    generated_sql_query = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_sql_query.strip()

# Define a function to compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits and labels to NumPy arrays if needed
    if isinstance(logits, tuple):
        logits = logits[0]  # Extracting the first element from the tuple
    if isinstance(labels, tuple):
        labels = labels[0]  # Extracting the first element from the tuple

    # Ensure logits and labels are numpy arrays and check their shapes
    logits = np.array(logits)
    labels = np.array(labels)

    try:
        predictions = np.argmax(logits, axis=-1)
        # Flatten the labels and predictions
        flat_labels = labels.flatten()
        flat_predictions = predictions.flatten()
    except ValueError as e:
        print(f"ValueError: {e}")
        return {}

    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_labels,
        flat_predictions,
        average='weighted',
        zero_division=1  # Set zero_division to handle warnings
    )
    acc = accuracy_score(flat_labels, flat_predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Load the dataset
full_dataset = pd.read_csv("train.csv")
data = full_dataset.sample(frac=0.1, random_state=42)

# Check if the dataset is empty
if data.empty:
    raise ValueError("The dataset is empty. Please check the dataset file.")

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.5, random_state=42)

# Verify Data Loading
print("Train Data Sample:")
print(train_data.head())

print("\nValidation Data Sample:")
print(val_data.head())

# Check dataset sizes after splitting
if len(train_data) == 0 or len(val_data) == 0:
    raise ValueError("One of the datasets after splitting is empty.")

# Function to preprocess the data
def preprocess_data(data):
    # Translate questions to English
    data['question'] = data['question'].apply(translate_question)
    inputs = tokenizer(data['question'].tolist(), max_length=512, truncation=True, padding='max_length', return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(data['sql'].tolist(), max_length=512, truncation=True, padding='max_length', return_tensors="pt")
    dataset = Dataset.from_dict({
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': labels['input_ids']
    })
    return dataset

# Preprocess the train and validation datasets
train_dataset = preprocess_data(train_data)
val_dataset = preprocess_data(val_data)

# Log dataset sizes after splitting
print(f"Initial Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Check if the output directory exists
output_dir = "./output"  # Unique identifier for each training run
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    eval_strategy="epoch",
    logging_dir="./logs",
    save_total_limit=5,
    num_train_epochs=1,
    load_best_model_at_end=True,
    save_strategy="epoch",
    report_to="tensorboard",  # Enable TensorBoard reporting
    remove_unused_columns=False
)

# Print training dataset size before training
print(f"Training dataset size before training: {len(train_dataset)}")

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
try:
    trainer.train()
except Exception as e:
    print("An error occurred during training:", e)
    # Log additional debug information
    if hasattr(trainer, '_signature_columns'):
        print(f"Model signature columns: {trainer._signature_columns}")
    # Print the keys of the first batch in the dataset for debugging
    for batch in trainer.get_train_dataloader():
        print(f"First batch keys: {batch.keys()}")
        break

model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')

# Evaluate the fine-tuned model
# Define sample questions and their corresponding SQL queries
sample_questions = [
    "What is the average salary of employees?",
    "Show all employees in the sales department.",
    "List all departments with more than 10 employees.",
    # Add more sample questions as needed
]

sample_sql_queries = [
    "SELECT AVG(salary) FROM employees;",
    "SELECT * FROM employees WHERE department = 'sales';",
    "SELECT department_name FROM departments WHERE employee_count > 10;",
    # Add corresponding SQL queries for sample questions
]

# Evaluate the fine-tuned model on sample questions
sample_eval_dataset = preprocess_data(pd.DataFrame({'question': sample_questions, 'sql': sample_sql_queries}))
sample_results = trainer.evaluate(eval_dataset=sample_eval_dataset)

# Check the keys in sample_results
print("Keys in sample_results:", sample_results.keys())

# Assuming predictions are stored under a key named 'predictions'
predictions = sample_results.get('predictions', None)

# Iterate over sample questions, predictions, and actual SQL queries
if predictions:
    for question, generated_query, actual_query in zip(sample_questions, predictions, sample_sql_queries):
        print(f"Question: {question}")
        print(f"Generated SQL Query: {generated_query}")
        print(f"Actual SQL Query: {actual_query}")
        print()
else:
    print("No predictions found in sample_results.")
