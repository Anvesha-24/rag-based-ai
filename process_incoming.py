import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

# Use the same model as stt.py for consistency
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_rag_query(incoming_query):
    if not os.path.exists('embeddings.joblib'):
        return "Database not found. Please process a lecture first."

    df = joblib.load('embeddings.joblib')
    
    # Search logic
    query_vector = embed_model.encode([incoming_query])
    similarities = cosine_similarity(np.vstack(df['embedding']), query_vector).flatten()
    top_indices = similarities.argsort()[::-1][:3]
    context = df.iloc[top_indices].to_json(orient="records")

    prompt = f"""
    Use these transcript chunks to answer the student. 
    Include timestamps in your answer.
    
    DATA: {context}
    QUESTION: {incoming_query}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error connecting to Groq: {e}"