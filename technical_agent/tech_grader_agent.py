import os
import json
from typing import Optional

import openai
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
assert api_key, "Missing OPENAI_API_KEY in environment"
openai.api_key = api_key

# Default config
DEFAULT_MODEL = "gpt-4o-mini"
# DEFAULT_MODEL = "gpt-4-0125-preview"
# DEFAULT_MODEL = "gpt-4o"

DEFAULT_TEMPERATURE = 0
DEFAULT_SEED = 42


def build_tech_grading_prompt(questions: str, answers: str, rubric: Optional[str]) -> dict:
    """
    Builds the system and user prompts for the LLM.
    """
    system_prompt = """
You are a grading assistant for technical exams. Your role is to evaluate student responses based on the provided questions and, if available, the rubric.

Grading Instructions:
- For each question:
  - Read the question and the student's answer.
  - If a rubric is provided for that question, follow it carefully to assign points based on the expected criteria.
  - If no rubric is provided, use your expert-level knowledge of the subject to assess:
    - Factual accuracy.
    - Completeness.
    - Clarity.
- Assign a score for each answer:
  - Use the point scale from the rubric, or if none is provided, use a default scale of 0-10 points.
- Provide feedback for each answer:
  - Explain why the student received the score.
  - Offer suggestions for improvement if the answer is incomplete or incorrect.

Respond ONLY with raw JSON (no markdown):
{
  "question_1": {
    "score": X,
    "feedback": "..."
  },
  "question_2": {
    "score": Y,
    "feedback": "..."
  },
  ...
  "total_score": Z
}
""".strip()

    user_prompt = f"""
### Rubric (if available):
{rubric or "No rubric provided."}

### Exam Questions:
{questions}

### Student Answers:
{answers}
""".strip()

    return {
        "system": system_prompt,
        "user": user_prompt
    }


def call_openai_chat(system: str, user: str) -> str:
    """
    Sends a message to OpenAI's chat completion API.
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
    Attempts to parse the LLM output into a JSON object.
    """
    if content.startswith("```"):
        content = content.split("```")[1].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Error: Could not parse JSON from LLM response.")
        print("Raw response:", content)
        return None


def grade_exam(questions_markdown: str, answers_text: str, rubric_markdown: Optional[str] = None) -> Optional[dict]:
    """
    Main grading function. Sends prompts to the LLM and parses the result.
    """
    prompts = build_tech_grading_prompt(questions_markdown, answers_text, rubric_markdown)
    raw_response = call_openai_chat(prompts["system"], prompts["user"])
    return parse_llm_response(raw_response)
