import os
import requests
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_API_URL")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

# Initialize FastAPI
app = FastAPI()

# CORS settings (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
dataset = load_dataset("Nitral-AI/Cybersecurity-ShareGPT", split="train")
conversation_data = [msg["value"]
                     for row in dataset
                     for msg in row["conversations"]
                     if msg["from"] == "human"]

# Encode with sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")
conversation_embeddings = model.encode(conversation_data)
print("✅ Embeddings loaded.")

# Input model
class UserQuery(BaseModel):
    message: str

# Similarity matcher
def get_most_relevant_response(query, threshold=0.6):
    query_embedding = model.encode([query])
    similarities = np.dot(conversation_embeddings, query_embedding.T).flatten()
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score < threshold:
        return None
    return conversation_data[best_idx]

# Chat route
@app.post("/chat/")
async def chat_with_bot(user_query: UserQuery):
    prompt = user_query.message.strip()
    if not prompt:
        return {"response": "Please type a message to continue."}

    similar_query = get_most_relevant_response(prompt)

    # Set clean system prompt
    if similar_query:
        system_prompt = (
            f"You are CyberAssist AI, a professional cybersecurity assistant. "
            f"Reply clearly and concisely (3-5 sentences max). Only include technical details if relevant.\n\n"
            f"User: {prompt}\n"
            f"Related past question: {similar_query}\n"
            f"Answer:"
        )
    else:
        system_prompt = (
            f"You are CyberAssist AI, a professional cybersecurity assistant. "
            f"Reply clearly and concisely (3-5 sentences max). Only include technical details if relevant.\n\n"
            f"User: {prompt}\n"
            f"Answer:"
        )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": system_prompt}]
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        bot_reply = response.json()["choices"][0]["message"]["content"]
        return {"response": bot_reply.strip() if bot_reply.strip() else "I'm not sure how to help with that. Try rephrasing."}
    except Exception as e:
        return {"response": f"⚠️ Error: {str(e)}"}


# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
