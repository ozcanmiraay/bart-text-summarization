from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model and tokenizer
model_path = "checkpoints/checkpoint-2500"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Initialize FastAPI app
app = FastAPI(title="Text Summarization API")

# Request schema
class SummarizationRequest(BaseModel):
    text: str

# Endpoint
@app.post("/summarize")
def summarize(request: SummarizationRequest):
    try:
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        summary_ids = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            no_repeat_ngram_size=3
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))