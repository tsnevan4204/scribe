FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg git && apt-get clean

RUN pip install --upgrade pip
RUN pip install faster-whisper torch numpy ffmpeg-python

WORKDIR /app
COPY . .

CMD ["python", "transcribe.py"]