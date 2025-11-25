import requests
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    # texts can be a string or a list of strings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"]
    return embedding

chunk_id = 0
my_dicts = []
jsons = os.listdir("jsons")

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")    
    embeddings = create_embedding([c['text'] for c in content['chunks']])    
    
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        # Minimal safety check to avoid IndexError
        if i < len(embeddings):
            chunk['embedding'] = embeddings[i]
        else:
            chunk['embedding'] = None
        chunk_id += 1
        my_dicts.append(chunk)
     
        
print(my_dicts)  

df = pd.DataFrame.from_records(my_dicts)     

joblib.dump(df, 'embeddings.joblib')

 