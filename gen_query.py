import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from googletrans import Translator

# Initialize the translator
translator = Translator()

# Load the trained model and tokenizer
model_name = "SwastikM/bart-large-nl2sql"
pre_trained_model_dir = "./trained_model"
tokenizer = BartTokenizer.from_pretrained(pre_trained_model_dir)
model = BartForConditionalGeneration.from_pretrained(pre_trained_model_dir)  # Replace with the path to your trained model

# Define a function to generate SQL query from English question using the trained model
def generate_sql_query(question):
    # Translate the question to English
    translated_question = translator.translate(question, src='pt', dest='en').text
    print(translated_question)
    # Generate SQL query from the translated question
    input_ids = tokenizer(translated_question, return_tensors="pt").input_ids  # No need to move input tensors to GPU
    generated_ids = model.generate(input_ids)
    generated_sql_query = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_sql_query.strip()

# Define sample questions
sample_questions = [
    "Mostre me todos os livros de camoes desde 1980",
    "Quem é o autor do livro os maias"
    # Add more sample questions as needed
]

# Generate SQL queries for sample questions
for question in sample_questions:
    generated_sql_query = generate_sql_query(question)
    print(f"Question: {question}")
    print(f"Generated SQL Query: {generated_sql_query}")
    print()
