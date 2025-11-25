import whisper
import json

model=whisper.load_model("base")

result=model.transcribe("audios/Astronomy I - 01.mp3",word_timestamps=False)

chunks=[]
for segment in result["segments"]:
    chunks.append({"start":segment["start"],"end": segment["end"],"text":segment["text"]})
    
    print(chunks)
    
    with open("output.json","w") as f:
        json.dump(chunks,f)