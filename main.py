from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BartTokenizer, BartForConditionalGeneration
from googletrans import Translator
import torch

# Initialize FastAPI app
app = FastAPI()

# Initialize the Translator
translator = Translator()

# Load the best model checkpoint
best_model_checkpoint = './output/checkpoint-42'
tokenizer = BartTokenizer.from_pretrained(best_model_checkpoint)
model = BartForConditionalGeneration.from_pretrained(best_model_checkpoint)

# Check if GPU is available and move model to GPU if it is
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the request model
class QueryRequest(BaseModel):
    question: str

# Translate question from Portuguese to English
def translate_question(question):
    translated_text = translator.translate(question, src='pt', dest='en')
    return translated_text.text

# Generate SQL query from English question
def generate_sql_query(question):
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(input_ids)
    generated_sql_query = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_sql_query.strip()

@app.post("/generate-sql/")
def generate_sql(request: QueryRequest):
    try:
        # Translate the question
        translated_question = translate_question(request.question)
        
        # Generate the SQL query
        generated_sql = generate_sql_query(translated_question)
        
        return {"generated_sql": generated_sql}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "NL2SQL working"}
