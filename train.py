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
        # Squeeze to ensure dimensions are [seq_length]
        source_inputs = {key: value.squeeze(0) for key, value in source_inputs.items()}
        target_inputs = {key: value.squeeze(0) for key, value in target_inputs.items()}
        return source_inputs, target_inputs

def collate_fn(batch):
    source_inputs = {key: torch.stack([item[0][key] for item in batch]) for key in batch[0][0].keys()}
    target_inputs = {key: torch.stack([item[1][key] for item in batch]) for key in batch[0][1].keys()}
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
        
        input_ids = source_inputs["input_ids"]
        attention_mask = source_inputs.get("attention_mask")
        labels = target_inputs["input_ids"]
        
        # Ensure correct dimensions
        assert input_ids.dim() == 2, f"Expected input_ids to be 2D, got {input_ids.dim()}D"
        if attention_mask is not None:
            assert attention_mask.dim() == 2, f"Expected attention_mask to be 2D, got {attention_mask.dim()}D"
        assert labels.dim() == 2, f"Expected labels to be 2D, got {labels.dim()}D"
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    return total_loss / len(dataloader)

# Prepare data
dataset = MyDataset('nl_marc.csv')
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Define optimizer and training parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

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
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs)
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
