import whisper
import json
import os
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load models once at the top level
stt_model = whisper.load_model("base")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def transcribe_and_embed(audio_path):
    # 1. Transcription
    result = stt_model.transcribe(audio_path, word_timestamps=False)
    
    chunks = []
    texts_only = []
    for segment in result["segments"]:
        chunk = {
            "title": os.path.basename(audio_path),
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        }
        chunks.append(chunk)
        texts_only.append(segment["text"])

    # 2. Generate Embeddings
    embeddings = embed_model.encode(texts_only)
    
    # 3. Create DataFrame and Save
    df = pd.DataFrame(chunks)
    df['embedding'] = list(embeddings)
    
    # Save for the RAG logic to use
    joblib.dump(df, 'embeddings.joblib')
    
    # 4. Save JSON for backup/reference
    base_name = os.path.basename(audio_path).split('.')[0]
    os.makedirs("jsons", exist_ok=True)
    with open(f"jsons/{base_name}.json", "w") as f:
        json.dump(chunks, f, indent=4)

    return result["text"]

if __name__ == "__main__":
    transcribe_and_embed("audios/Astronomy I - 01.mp3")