import os
import json
from typing import Optional

import openai
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "Missing OPENAI_API_KEY in environment"
openai.api_key = api_key

# Config
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_SEED = 42


def build_narrative_prompt(questions: str, responses: str, rubric: Optional[str]) -> dict:
    """
    Build the system and user prompt for narrative exam grading.
    """
    if rubric and rubric.strip():
        system_prompt = (
            "You are an exam grader. Use the rubric to assign each question a numeric score (0-10) "
            "and provide clear, concise feedback to help the student understand their strengths and weaknesses. "
            "Then calculate the total score and return everything as valid JSON."
        )
        user_prompt = f"Rubric:\n{rubric}\n\nQuestions:\n{questions}\n\nStudent Responses:\n{responses}"
    else:
        system_prompt = (
            "You are an exam grader. No rubric is available. Use your expert judgment to assign each question "
            "a numeric score (0-10) and provide useful feedback. Then compute the total score. Return as valid JSON."
        )
        user_prompt = f"Questions:\n{questions}\n\nStudent Responses:\n{responses}"

    return {
        "system": system_prompt,
        "user": user_prompt
    }


def call_openai_chat(system: str, user: str) -> str:
    """
    Calls OpenAI's chat API with the given prompts.
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
    Parses JSON response content safely.
    """
    if content.startswith("```"):
        content = content.split("```")[1].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON from LLM response.")
        print("Raw response:", content)
        return None


def grade_exam(questions: str, responses: str, rubric: Optional[str] = None) -> Optional[dict]:
    """
    Main grading function for narrative exams.
    """
    prompts = build_narrative_prompt(questions, responses, rubric)
    raw_response = call_openai_chat(prompts["system"], prompts["user"])
    return parse_llm_response(raw_response)
