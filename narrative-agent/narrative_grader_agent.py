import os
from pathlib import Path
import time
import json
import statistics
import pickle

from dotenv import load_dotenv
import openai
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from openai import OpenAIError, RateLimitError, APIConnectionError, Timeout

# Load environment variables from .env (if present)
load_dotenv()

# Configuration: read API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0  # seconds

# Retry wrapper with exponential backoff
def call_with_backoff(**kwargs):
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return openai.chat.completions.create(**kwargs)
        except (RateLimitError, OpenAIError):
            if attempt == MAX_RETRIES:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))
        except (APIConnectionError, Timeout):
            if attempt == MAX_RETRIES:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))

# Grade the exam via OpenAI
def grade_exam(rubric: str, questions: str, responses: str) -> dict:
    start_time = time.time()

    if rubric.strip():
        system_prompt = (
            "You are an exam grader. Use the rubric to assign each question a numeric score (0-10) and valuable concise feedback so the student can further understand their strengths and weaknesses of the material. "
            "Then compute the overall score as the average and provide general feedback. Return JSON."
        )
        user_prompt = f"Rubric:\n{rubric}\n\nQuestions:\n{questions}\n\nStudent Responses:\n{responses}"
    else:
        system_prompt = (
            "You are an exam grader. The rubric is not available. Use your own criteria to assign each question a numeric score (0-10) and constructive feedback. "
            "Then compute the overall score as the average and provide general feedback. Return JSON."
        )
        user_prompt = f"Questions:\n{questions}\n\nStudent Responses:\n{responses}"

    resp = call_with_backoff(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        seed=42
    )

    output = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
    return output