# Text Summarization with BART and MLflow

## Overview

This project demonstrates an end-to-end **text summarization** pipeline using the [Hugging Face Transformers](https://github.com/huggingface/transformers) library. We train a `facebook/bart-base` model on the [CNN/DailyMail dataset](https://huggingface.co/datasets/cnn_dailymail) and log training/evaluation metrics with [MLflow](https://mlflow.org/). The code is designed to be **MLOps-ready**, showcasing experiment tracking, reproducibility, and deployment readiness.

---

## Project Structure

```
text_summarization_project/
├── checkpoints/                # Model checkpoints saved during training
├── mlruns/                     # MLflow run logs (auto-created)
├── src/
│   ├── __init__.py
│   ├── train.py                # Main training script
│   ├── app.py                  # FastAPI app exposing /summarize endpoint
│   └── generate_examples.py    # Inference script to print sample summaries
├── venv/                       # (Optional) local virtual environment
├── README.md                   # This README
└── requirements.txt            # Python dependencies
```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YourUsername/text_summarization_project.git
   cd text_summarization_project
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or venv\Scripts\activate on Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

> **Note**: Ensure you have Python 3.9+ installed. On Apple Silicon (M1/M2/M3) Macs, the model can use Metal Performance Shaders (MPS) for GPU acceleration.

---

## Usage

### 1. Train the Model

From the root folder:

```bash
python src/train.py --epochs 1 --batch_size 2
```

Trained models are saved in `checkpoints/`.

---

### 2. View MLflow Logs

In a separate terminal tab:

```bash
mlflow ui
```

Then visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

### 3. Sample Predictions Locally

To print summaries from your trained model:

```bash
python src/generate_examples.py
```

This will display a few CNN/DailyMail articles with:
- The **original article (truncated)**
- The **reference summary**
- The **predicted summary**

---

### 4. Run FastAPI Server (Deploy Model)

Start the FastAPI server:

```bash
python -m uvicorn src.app:app --reload --port 8000
```

Then go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to try the `/summarize` endpoint interactively.

You can also test it via cURL:

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your article text here..."}'
```

---

## Example: POST to `/summarize`

**Input:**
```json
{
  "text": "President Obama held a press conference today..."
}
```

**Response:**
```json
{
  "summary": "Obama held a press conference to address the issue."
}
```

---

## Next Steps

- **Compute ROUGE scores** for evaluation
- **Enable monitoring** for drift and retraining
- **Integrate with containerization tools** like Docker for deployment
- **Upload model to Hugging Face Hub** for sharing and reproducibility

---

## Contact

For questions or suggestions, please contact:

**Miray Özcan**  
`miray@uni.minerva.edu`