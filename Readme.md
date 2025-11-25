#how to use this RAG AI teaching assistant on your own data
## step 1-collect your videos
Move all your video files to the video folder

## step 2-Convert to mp3
convert all the video files to mp3 by running video_to_mp3

## step 3-CONVERT MP3 TO JSON
Convert all the mp3 files to json by running mp3_to_json

## step 4 -convert the json files to vector
use the file preprocess_json to convert the json files to a dataframe with embeddings and save it as a joblib pickle

## step 5-Prompt generation and feeding to LLM
Read the joblib file and load it into the memory.then create a relevant prompt as per the user query and feed it to the LLM

