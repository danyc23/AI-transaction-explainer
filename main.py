from fastapi import FastAPI, UploadFile
import pandas as pd
import openai
from openai import RateLimitError
from fastapi import HTTPException

client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # Ollama-compatible
app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile):
    df = pd.read_csv(file.file)
    total = df["amount"].sum()
    categories = df.groupby("category")["amount"].sum().to_dict()
    prompt = f"You are a financial assistant. Here's a breakdown of user spending:\n\nTotal: ${total}\nCategories: {categories}\n\nSummarize insights in natural language."
    try:
        response = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        )
        print(response.choices[0].message.content)
        return {"summary": response.choices[0].message.content}
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail="Rate limit exceeded or quota exhausted.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error from model: {str(e)}")