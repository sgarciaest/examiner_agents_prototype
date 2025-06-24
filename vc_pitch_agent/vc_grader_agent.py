import os
import json
from typing import Optional, Tuple

import openai
import librosa
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "Missing OPENAI_API_KEY in environment"
openai.api_key = api_key

# Config
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0
DEFAULT_SEED = 42
CACHE_DIR = os.path.join(os.getcwd(), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# VC grading rubric
RUBRIC_TEMPLATE = """
You are a seasoned VC pitch grader. For a {duration:.1f}-minute audio pitch, give each dimension a score from 1 (poor) to 10 (excellent), using the following anchors:

1. Problem Clarity  
   • 1–3: No clear problem stated, listener confused  
   • 4–6: Problem mentioned but lacks context or urgency  
   • 7–8: Problem clearly described with context  
   • 9–10: Problem statement is crisp, impactful, and immediately compelling

2. Market Evidence  
   • 1–3: No market data or vague claims  
   • 4–6: Qualitative market description, no numbers  
   • 7–8: One clear quantitative metric (TAM, growth rate)  
   • 9–10: Multiple strong data points (TAM, traction, growth) cited

3. Solution Differentiation  
   • 1–3: Solution not differentiated, generic  
   • 4–6: Mentions a unique feature but no defense  
   • 7–8: Clearly highlights one defensible advantage  
   • 9–10: Demonstrates multiple, well-justified differentiators or proprietary edge

4. Delivery & Pacing  
   • 1–3: Monotone or too fast/slow (outside 80–200 WPM), frequent long pauses (>30 %)  
   • 4–6: Understandable but some pacing issues (WPM 90–210, pauses 20–30 %)  
   • 7–8: Good pace (110–160 WPM), pauses <20 %  
   • 9–10: Engaging tone, ideal pacing (120–150 WPM), minimal pauses (<10 %)

Return valid JSON EXACTLY in this format (no extra keys):
{{
  "Problem": <1–10>,
  "Market": <1–10>,
  "Solution": <1–10>,
  "Delivery": <1–10>,
  "Feedback": "<one sentence actionable feedback for each anchor>"
}}
"""


def transcribe_audio(mp3_path: str) -> str:
    """
    Transcribes the MP3 using Whisper and caches the result.
    """
    cache_file = os.path.join(CACHE_DIR, os.path.basename(mp3_path) + ".txt")
    if os.path.exists(cache_file):
        return open(cache_file).read()

    with open(mp3_path, "rb") as f:
        response = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )
    transcript = response if isinstance(response, str) else response.get("text", "")
    with open(cache_file, "w") as f:
        f.write(transcript)

    return transcript


def analyze_audio(mp3_path: str) -> Tuple[float, float, str, float]:
    """
    Returns WPM, silence ratio, transcript, and duration.
    """
    y, sr = librosa.load(mp3_path, sr=16000, mono=True)
    duration = len(y) / sr

    transcript = transcribe_audio(mp3_path)
    words = transcript.split()
    wpm = len(words) / (duration / 60) if duration else 0

    segments = librosa.effects.split(y, top_db=30)
    voiced = sum((end - start) for start, end in segments) / sr
    silence_ratio = (duration - voiced) / duration if duration else 0

    return wpm, silence_ratio, transcript, duration


def build_vc_prompt(transcript: str, wpm: float, silence: float, duration: float) -> str:
    """
    Composes the full prompt with transcript and metrics.
    """
    rubric = RUBRIC_TEMPLATE.format(duration=duration / 60)
    return f"""
Pitch transcript:
{transcript}

Audio metrics:
• Words-per-minute: {wpm:.1f}
• Pause ratio: {silence:.1%}

{rubric}
""".strip()


def call_openai_chat(system: str, user: str) -> str:
    """
    Sends the prompt to OpenAI and returns the raw response.
    """
    response = openai.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=DEFAULT_TEMPERATURE,
        seed=DEFAULT_SEED
    )
    return response.choices[0].message.content


def parse_llm_response(content: str) -> Optional[dict]:
    """
    Parses JSON response from the LLM.
    """
    content = content.replace("\n", " ").strip()
    if content.startswith("```"):
        content = content.split("```")[1].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Error: Could not parse LLM response.")
        print("Raw response:", content)
        return None


def grade_pitch(mp3_path: str) -> Optional[dict]:
    """
    Main grading function for VC pitches.
    """
    wpm, silence, transcript, duration = analyze_audio(mp3_path)
    prompt = build_vc_prompt(transcript, wpm, silence, duration)
    raw_response = call_openai_chat("You are a helpful pitch grader.", prompt)
    return parse_llm_response(raw_response)
