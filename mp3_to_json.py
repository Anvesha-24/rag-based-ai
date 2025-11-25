import whisper
import json
import os

#load model
model=whisper.load_model("base")

#list audio files
audios=os.listdir("audios")

for audio in audios:
    print("processing:",audio)
    title=audio.split("-")[0]
    print("Title:",title)
    
    audio_path=f"audios/{audio}"
    
    result=model.transcribe(audio=audio_path,word_timestamps=False)
    
    chunks=[]
    
    for segment in result["segments"]:
        chunks.append({"title":title,
                       "start":segment["start"],
                       "end":segment["end"],
                       "text":segment["text"]
                       })
        
    chunks_with_metadata={"chunks":chunks,"text":result["text"]}
    
    with open(f"jsons/{audio}.json","w") as f:
        json.dump(chunks_with_metadata,f)
