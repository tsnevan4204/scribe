from faster_whisper import WhisperModel
import torch

print(torch.__version__)
print(torch.cuda.is_available())

model = WhisperModel("tiny", compute_type="int8")
segments, info = model.transcribe("sample.wav")

print("Detected language:", info.language)
for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")