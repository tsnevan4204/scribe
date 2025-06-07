# === Updated Flask App with Better Audio Handling ===
import litellm
from litellm import completion
from flask import Flask, request, send_from_directory, jsonify
from google.cloud import storage, speech_v1p1beta1 as speech
from dotenv import load_dotenv
from datetime import datetime
import os, logging, ffmpeg, wave, tempfile, shutil
import pytz

load_dotenv()
logging.basicConfig(level=logging.INFO)

litellm.api_key = os.getenv("GROQ_API_KEY")
litellm.provider = "groq"
MODEL_NAME="groq/deepseek-r1-distill-llama-70b"

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Google Cloud
gcs_bucket = os.getenv("GCS_BUCKET_NAME")
if not gcs_bucket:
    raise ValueError("GCS_BUCKET_NAME not set")

client = storage.Client()
bucket = client.bucket(gcs_bucket)

# Session state
SESSION_ACTIVE = False
SESSION_FOLDER = ""
TRANSCRIPT_PATH = ""
EST = pytz.timezone("US/Eastern")

# Util: Get timestamp
def est_now():
    return datetime.now(EST).strftime("%Y%m%d_%H%M%S")

# Util: Extract sample rate
def get_sample_rate(wav_path):
    with wave.open(wav_path, 'rb') as wf:
        return wf.getframerate()

# Util: Convert WebM to WAV with better error handling
def convert_webm_to_wav(webm_path, wav_path):
    try:
        # Check if input file exists and has content
        if not os.path.exists(webm_path):
            raise ValueError(f"Input file {webm_path} does not exist")
        
        file_size = os.path.getsize(webm_path)
        if file_size == 0:
            raise ValueError(f"Input file {webm_path} is empty")
        
        logging.info(f"Converting WebM file: {webm_path} (size: {file_size} bytes)")
        
        # Use ffmpeg with more robust settings
        (
            ffmpeg
            .input(webm_path)
            .output(
                wav_path,
                format='wav',
                acodec='pcm_s16le',
                ar=16000,  # Standard sample rate for speech recognition
                ac=1,      # Mono audio
                loglevel='error'  # Reduce ffmpeg verbosity
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        # Verify the output file was created successfully
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            raise ValueError("Failed to create valid WAV file")
            
        logging.info(f"✅ Successfully converted {webm_path} to {wav_path}")
        
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logging.error(f"❌ FFmpeg error: {error_msg}")
        raise ValueError(f"Audio conversion failed: {error_msg}")
    except Exception as e:
        logging.error(f"❌ Conversion error: {str(e)}")
        raise

# === ROUTES ===

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/start_session", methods=["POST"])
def start_session():
    global SESSION_ACTIVE, SESSION_FOLDER, TRANSCRIPT_PATH
    if SESSION_ACTIVE:
        return {"error": "Session already active"}, 400

    data = request.get_json()
    raw_name = data.get("session_name", "").strip()

    if not raw_name or "/" in raw_name or "\\" in raw_name:
        return {"error": "Invalid session name"}, 400

    SESSION_ACTIVE = True
    session_name = f"session_{raw_name}"
    SESSION_FOLDER = os.path.join(UPLOAD_FOLDER, session_name)
    os.makedirs(SESSION_FOLDER, exist_ok=True)
    TRANSCRIPT_PATH = os.path.join(SESSION_FOLDER, f"transcript_{session_name}.txt")
    logging.info(f"🟢 Session started in {SESSION_FOLDER}")
    return {"message": "Session started"}, 200

@app.route("/end_session", methods=["POST"])
def end_session():
    global SESSION_ACTIVE, SESSION_FOLDER, TRANSCRIPT_PATH
    if not SESSION_ACTIVE:
        return {"error": "No session active"}, 400

    try:
        session_name = SESSION_FOLDER.split('/')[-1]

        # Upload transcript to GCS if exists
        if os.path.exists(TRANSCRIPT_PATH):
            transcript_blob = bucket.blob(f"{session_name}/transcript_{session_name}.txt")
            transcript_blob.upload_from_filename(TRANSCRIPT_PATH, content_type="text/plain")
            logging.info(f"☁️ Final transcript upload to GCS: {session_name}/transcript_{session_name}.txt")

            # === Run LLM on transcript ===
            with open(TRANSCRIPT_PATH, "r", encoding='utf-8') as f:
                transcript_text = f.read()

            medical_record = generate_medical_record_from_transcript(transcript_text)

            # Save locally
            structured_path = os.path.join(SESSION_FOLDER, f"structured_{session_name}.txt")
            with open(structured_path, "w", encoding='utf-8') as f:
                f.write(medical_record)
            logging.info(f"📝 Structured medical record saved: {structured_path}")

            # Upload to GCS
            blob_structured = bucket.blob(f"{session_name}/structured_{session_name}.txt")
            blob_structured.upload_from_filename(structured_path, content_type="text/plain")
            logging.info(f"☁️ Uploaded structured note to GCS: {session_name}/structured_{session_name}.txt")

        # Clean up local folder
        if os.path.exists(SESSION_FOLDER):
            shutil.rmtree(SESSION_FOLDER)
            logging.info(f"🧹 Cleaned up local session folder: {SESSION_FOLDER}")

        cleanup_residual_folders()
        cleanup_gcs_residual_folders()

    except Exception as cleanup_error:
        logging.warning(f"⚠️ Session cleanup warning: {cleanup_error}")

    SESSION_ACTIVE = False
    logging.info(f"🔴 Session ended")
    return {"message": "Session ended"}, 200

def cleanup_residual_folders():
    """Clean up any folders from old naming patterns"""
    import glob
    
    # Look for old pattern folders (YYYYMMDD_HHMMSS without 'session_' prefix)
    old_pattern_folders = glob.glob("20*_*")
    for folder in old_pattern_folders:
        if os.path.isdir(folder) and folder.count('_') == 1:  # YYYYMMDD_HHMMSS pattern
            try:
                # Only clean up if empty or contains only empty subdirs
                if not os.listdir(folder) or all(not os.listdir(os.path.join(folder, item)) 
                                                for item in os.listdir(folder) 
                                                if os.path.isdir(os.path.join(folder, item))):
                    shutil.rmtree(folder)
                    logging.info(f"🧹 Cleaned up residual folder: {folder}")
            except Exception as e:
                logging.warning(f"⚠️ Could not clean up residual folder {folder}: {e}")

import re

def clean_llm_output(text: str) -> str:
    """Removes DeepSeek-style <think>...</think> blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def generate_medical_record_from_transcript(transcript_text: str) -> str:
    prompt = (
        "You are a clinical documentation assistant. Format the following unstructured "
        "medical transcript into a structured SOAP medical record.\n\nTranscript:\n"
        f"{transcript_text.strip()}"
    )

    try:
        result = completion(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a medical record assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        raw_output = result["choices"][0]["message"]["content"]
        return clean_llm_output(raw_output)
    except Exception as e:
        return f"[Error generating medical record: {str(e)}]"

def cleanup_gcs_residual_folders():
    """Clean up empty folders in GCS bucket"""
    try:
        # List all blobs to find folder patterns
        blobs = list(bucket.list_blobs())
        
        # Group by folder structure
        folders = {}
        for blob in blobs:
            parts = blob.name.split('/')
            if len(parts) >= 2:
                folder = parts[0]
                if folder not in folders:
                    folders[folder] = []
                folders[folder].append(blob)
        
        # Find empty folders or folders with old naming patterns
        for folder_name, folder_blobs in folders.items():
            # Check for old naming pattern (YYYYMMDD_HHMMSS without session_ prefix)
            if folder_name.count('_') == 1 and folder_name.startswith('202') and not folder_name.startswith('session_'):
                if len(folder_blobs) == 0:
                    logging.info(f"🧹 Found empty residual GCS folder: {folder_name}")
                    # Note: GCS doesn't have actual empty folders, they're just prefixes
                    # The empty folder will disappear naturally when no objects use that prefix
        
    except Exception as e:
        logging.warning(f"⚠️ GCS cleanup warning: {e}")

@app.route("/upload", methods=["POST"])
def upload():
    if not SESSION_ACTIVE:
        return {"error": "No session active"}, 400

    try:
        logging.info("📥 Received upload request")
        audio_data = request.data
        
        if len(audio_data) == 0:
            logging.error("❌ Empty audio data received")
            return {"error": "Empty audio data"}, 400
        
        if len(audio_data) < 100:  # Very small files are likely corrupted
            logging.error(f"❌ Audio data too small: {len(audio_data)} bytes")
            return {"error": f"Audio data too small ({len(audio_data)} bytes). Please ensure microphone is working and speak for at least 1 second."}, 400
        
        logging.info(f"📊 Audio data size: {len(audio_data)} bytes")
        
        timestamp = est_now()
        base_filename = f"recording_{timestamp}"
        
        # Use temporary files for better handling
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as webm_temp:
            webm_temp.write(audio_data)
            webm_path = webm_temp.name
        
        wav_path = os.path.join(SESSION_FOLDER, f"{base_filename}.wav")
        
        try:
            # Convert to .wav
            convert_webm_to_wav(webm_path, wav_path)
            
            # Verify WAV file integrity
            sample_rate = get_sample_rate(wav_path)
            logging.info(f"📊 WAV file created: sample rate {sample_rate}Hz")
            
            # Upload .wav to GCS (keep recordings in GCS)
            wav_blob = bucket.blob(f"{SESSION_FOLDER.split('/')[-1]}/{base_filename}.wav")
            wav_blob.upload_from_filename(wav_path, content_type="audio/wav")
            logging.info(f"☁️ Uploaded .wav to GCS: {SESSION_FOLDER.split('/')[-1]}/{base_filename}.wav")

            # Transcribe
            audio_config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                language_code="en-US",
                enable_word_time_offsets=False,
                enable_automatic_punctuation=True,
            )

            with open(wav_path, "rb") as audio_file:
                audio_data = speech.RecognitionAudio(content=audio_file.read())

            response = speech.SpeechClient().recognize(config=audio_config, audio=audio_data)

            if response.results:
                transcript_parts = []
                for result in response.results:
                    if result.alternatives:
                        transcript_parts.append(result.alternatives[0].transcript)
                
                text = " ".join(transcript_parts).strip()
                
                if text:
                    # Append to local transcript file
                    transcript_entry = f"[{timestamp} EST] {text}\n"
                    with open(TRANSCRIPT_PATH, "a", encoding='utf-8') as f:
                        f.write(transcript_entry)
                    
                    # IMMEDIATELY upload updated transcript to GCS (backup after each recording)
                    try:
                        session_name = SESSION_FOLDER.split('/')[-1]  # Extract session name from path
                        transcript_blob = bucket.blob(f"{session_name}/transcript_{session_name}.txt")
                        transcript_blob.upload_from_filename(TRANSCRIPT_PATH, content_type="text/plain")
                        logging.info(f"☁️ Updated transcript in GCS: {session_name}/transcript_{session_name}.txt")
                    except Exception as transcript_upload_error:
                        logging.warning(f"⚠️ Failed to upload transcript: {transcript_upload_error}")
                    
                    logging.info(f"📝 Transcript: {text}")
                    return {"transcript": text, "message": "Transcription complete"}, 200
                else:
                    logging.info("🔇 No speech detected in audio")
                    return {"transcript": "", "message": "No speech detected"}, 200
            else:
                logging.info("🔇 No transcription results")
                return {"transcript": "", "message": "No speech detected"}, 200

        except Exception as conversion_error:
            logging.error(f"❌ Processing failed: {str(conversion_error)}")
            return {"error": f"Audio processing failed: {str(conversion_error)}"}, 500
        
        finally:
            # Cleanup temporary files (keep WAV in session folder until session ends)
            try:
                if os.path.exists(webm_path):
                    os.remove(webm_path)
                    logging.info("🧹 Cleaned up temporary WebM file")
                # Note: WAV file stays in session folder until end_session
            except Exception as cleanup_error:
                logging.warning(f"⚠️ Cleanup warning: {cleanup_error}")

    except Exception as e:
        logging.error(f"❌ Upload failed: {str(e)}")
        return {"error": f"Upload failed: {str(e)}"}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)