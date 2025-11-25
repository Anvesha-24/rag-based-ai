import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    try:
        embedding = r.json().get("embeddings")
    except:
        print("Embedding API returned invalid JSON:", r.text)
        return None
    return embedding


def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    
    try:
        data = r.json()
    except:
        print("Generate API returned invalid JSON:", r.text)
        return None
    
    print(data)
    return data


df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a Question: ")

question_embedding = create_embedding([incoming_query])
if question_embedding is None:
    print("Failed to create embedding. Exiting.")
    exit()

# find cosine similarities
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding[0]]).flatten()
print(similarities)

max_idx = (similarities.argsort()[::-1][0:3])
print(max_idx)
new_df = df.iloc[max_idx]  # FIXED: use iloc instead of loc

prompt = f'''
I am teaching astronomy. Here are the most relevant video subtitle chunks, each with
title, start time, end time, and text:

{new_df[["title", "start", "end", "text"]].to_json(orient="records")}

-------------------------------------------------
User query: "{incoming_query}"

You must answer:
1. Where in the video this topic is taught (timestamps)
2. How much content is taught there
3. Guide the student to the correct timestamp
If the question is unrelated, say:
"I can only answer questions related to the astronomy course."
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

res = inference(prompt)

if res is None or "response" not in res:
    print("Model returned no valid response. Full output:", res)
    exit()

response = res["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)
