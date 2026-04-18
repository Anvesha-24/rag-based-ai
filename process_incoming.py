import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq
import joblib
import os
import streamlit as st
from dotenv import load_dotenv

# 1. Setup API Key (Cloud-First Logic)
load_dotenv()
# If in cloud, use st.secrets. If local, use .env
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

# 2. Initialize Models
# We use the same 'all-MiniLM-L6-v2' as in stt.py for consistency
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=api_key)

def run_rag_query(incoming_query):
    """
    Search the indexed lecture data and get an answer from Groq.
    """
    # Check if the database exists
    if not os.path.exists('embeddings.joblib'):
        return "⚠️ No lecture data found. Please upload and process a lecture first!"

    try:
        # 3. Load the indexed data
        df = joblib.load('embeddings.joblib')

        # 4. Generate query embedding
        query_vector = embed_model.encode([incoming_query])

        # 5. Calculate similarity
        # We compare the query to all stored transcript embeddings
        similarities = cosine_similarity(np.vstack(df['embedding']), query_vector).flatten()
        
        # Get the top 3 most relevant segments
        top_indices = similarities.argsort()[::-1][:3]
        context_data = df.iloc[top_indices]
        
        # Format context for the LLM
        context_text = context_data[["start", "end", "text"]].to_json(orient="records")

        # 6. Create the Professional RAG Prompt
        prompt = f"""
        You are an AI Teaching Assistant for an Astronomy course. 
        Answer the student's question based ONLY on the following transcript segments.
        
        CONTEXT FROM LECTURE:
        {context_text}
        
        STUDENT QUESTION: 
        "{incoming_query}"
        
        RESPONSE GUIDELINES:
        - Be encouraging and educational.
        - If the answer is in the context, cite the timestamps (e.g., [12:30 - 13:45]).
        - If the answer is NOT in the context, politely explain that the lecture didn't cover that specific point.
        """

        # 7. Call Groq Llama 3.3 (Fastest & most accurate for RAG)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2, # Keeps the AI focused on the facts
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"❌ Error in RAG Pipeline: {str(e)}"

if __name__ == "__main__":
    # Quick terminal test
    test_query = "What is the name of the professor?"
    print(run_rag_query(test_query))