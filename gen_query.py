import torch
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the trained model and tokenizer
model_name = "SwastikM/bart-large-nl2sql"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained("./output/checkpoint")  # Replace "trained_model" with the path to your trained model

# Define a function to generate SQL query from English question using the trained model
def generate_sql_query(question):
    input_text = question
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids  # No need to move input tensors to GPU
    generated_ids = model.generate(input_ids)
    generated_sql_query = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_sql_query.strip()

# Define sample questions
sample_questions = [
    "What is the average salary of employees?",
    "Show all employees in the sales department.",
    "List all departments with more than 10 employees.",
    "Quem é o autor do livro os maias"
    # Add more sample questions as needed
]

# Generate SQL queries for sample questions
for question in sample_questions:
    generated_sql_query = generate_sql_query(question)
    print(f"Question: {question}")
    print(f"Generated SQL Query: {generated_sql_query}")
    print()
