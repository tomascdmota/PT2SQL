from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch
from torch.utils.data import DataLoader, Dataset
import random
import csv
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)

# Define dataset class with padding
class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                source_text, target_text = row[0], row[1]
                self.data.append((source_text, target_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text, target_text = self.data[idx]
        source_inputs = tokenizer(source_text, return_tensors="pt", padding="max_length", truncation=True, max_length=50)
        target_inputs = tokenizer(target_text, return_tensors="pt", padding="max_length", truncation=True, max_length=50)
        return source_inputs, target_inputs

# Define training function
def train_epoch(model, optimizer, dataloader, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        source_inputs, target_inputs = batch
        source_inputs = {key: value.to(device) for key, value in source_inputs.items()}
        target_inputs = {key: value.to(device) for key, value in target_inputs.items()}
        
        # Access input_ids from target_inputs
        labels = target_inputs["input_ids"]
        
        # Ensure attention mask has correct shape
        attention_mask = target_inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.float()  # Ensure it's a float tensor
            attention_mask = attention_mask.to(device)
        
        outputs = model(**source_inputs, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)

# Prepare data
dataset = MyDataset('nl_marc.csv')
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define optimizer and training parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 5

# Fine-tuning loop
for epoch in range(num_epochs):
    train_loss = train_epoch(model, optimizer, train_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

# Save fine-tuned model
model.save_pretrained("fine_tuned_bart_model")
tokenizer.save_pretrained("fine_tuned_bart_model")

# Function to generate MARC records for given Portuguese queries
def generate_marc_from_queries(queries, model, tokenizer, device):
    marc_records = []
    for query in queries:
        input_ids = tokenizer(query, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        attention_mask = tokenizer(query, return_tensors="pt", padding=True, truncation=True).attention_mask.to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        marc_records.append(decoded_output)
    return marc_records

# Example usage:
portuguese_queries = [
    "Quais são os livros escritos por Charles Dickens?",
    "Mostre-me todos os livros sobre a história do Japão.",
    "Encontre todos os livros sobre teoria dos jogos.",
    "Mostre-me os livros publicados em 1980."
]
marc_records = generate_marc_from_queries(portuguese_queries, model, tokenizer, device)
for query, marc_record in zip(portuguese_queries, marc_records):
    print(f"Portuguese Query: {query}\nGenerated MARC Record: {marc_record}\n")
