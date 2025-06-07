import litellm
from litellm import completion
import re
import dotenv
import logging
import os

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

litellm.api_key = os.getenv("GROQ_API_KEY")
litellm.provider = "groq"
MODEL_NAME="groq/deepseek-r1-distill-llama-70b"


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
