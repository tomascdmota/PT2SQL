import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, load_metric
from googletrans import Translator
from langdetect import detect
import os

# Load the pre-trained model
model_path = "./trained_model"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# Load the dataset
dataset = pd.read_csv("portuguese_nl_to_sql_dataset.csv")

translator = Translator()
# Check if GPU is available and move model to GPU if it is
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Model is using device: {model.device}")

def translate_question(question):
    try:
        if detect(question) == 'en':
            return question  # Return the question unchanged if it's already in English
        else:
            translated_text = translator.translate(question, src='pt', dest='en')
            return translated_text.text
    except Exception as e:
        print(f"Translation error: {e}")
        return question

# Preprocess the data
def preprocess_data(data):
    # Translate questions to English
    data['question'] = data['question'].apply(translate_question)
    inputs = tokenizer(data['question'].tolist(), max_length=512, truncation=True, padding='max_length', return_tensors="pt")
    labels = tokenizer(data['sql'].tolist(), max_length=512, truncation=True, padding='max_length', return_tensors="pt", text_target=data['sql'].tolist())
    dataset = Dataset.from_dict({
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': labels['input_ids']
    })
    return dataset

# Preprocess the dataset
processed_data = preprocess_data(dataset)

# Split the dataset into training and validation sets
train_test_split = processed_data.train_test_split(test_size=0.2, seed=42)
train_data = train_test_split['train']
val_data = train_test_split['test']

output_dir = "./output"  # Unique identifier for each training run
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
    evaluation_strategy="epoch",
    report_to="none",
    output_dir='./output'
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Fine-tune the model
trainer.train()

# Evaluate the fine-tuned model
results = trainer.evaluate()

# Save the fine-tuned model
trainer.save_model("fine_tuned_model")

print("Training and evaluation complete.")
print(f"Results: {results}")
