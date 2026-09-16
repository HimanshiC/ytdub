from transformers import pipeline
import time

AUDIO = r"E:\ytdub\.cache\ttespqxQbio\media\source_audio.webm"

started = time.perf_counter()

pipe = pipeline(
    "automatic-speech-recognition",
    model="ai4bharat/indicwav2vec_v2_hindi",
    device=-1,
)

result = pipe(AUDIO)

elapsed = time.perf_counter() - started

print("TIME:", elapsed)
print("TEXT:", result["text"])