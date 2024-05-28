import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# Load the pre-trained model
model_path = "./trained_model"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# Load the dataset
dataset = pd.read_csv("portuguese_nl_to_sql_dataset.csv")

# Preprocess the data
# Assume you have a function preprocess_data() for this purpose
processed_data = preprocess_data(dataset)

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(processed_data, test_size=0.2, random_state=42)

# Tokenize the data
train_encodings = tokenizer(train_data["prompt"], truncation=True, padding=True)
val_encodings = tokenizer(val_data["prompt"], truncation=True, padding=True)

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
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=val_encodings,
)

# Fine-tune the model
trainer.train()

# Evaluate the fine-tuned model
results = trainer.evaluate()

# Save the fine-tuned model
trainer.save_model("fine_tuned_model")
