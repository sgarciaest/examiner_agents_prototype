import os
import argparse
import json
from pathlib import Path

from utils import extract_pdf_to_markdown

# Import agents
from technical_agent.tech_grader_agent import grade_exam as grade_tech
from narrative_agent.narrative_grader_agent import grade_exam as grade_narrative
from vc_pitch_agent.vc_grader_agent import grade_pitch as grade_vc



def load_text_file(file_path: Path) -> str:
    """Load plain text from .txt or .md file."""
    return Path(file_path).read_text(encoding="utf-8")


def load_pdf_as_markdown(file_path: Path) -> str:
    """Convert PDF content into markdown using util function."""
    return extract_pdf_to_markdown(str(file_path))


def detect_input_type(file_path: Path) -> str:
    """Helper to infer content type from file extension."""
    ext = file_path.suffix.lower()
    if ext in [".txt", ".md"]:
        return "text"
    elif ext == ".pdf":
        return "pdf"
    elif ext in [".mp3", ".wav", ".m4a"]:
        return "audio"
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def run_tech_or_narrative(agent_type: str, questions_file, answers_file, rubric_file=None):
    """Run either the tech or narrative grading agent."""
    # Load content
    questions = load_pdf_as_markdown(questions_file) if questions_file.suffix == ".pdf" else load_text_file(questions_file)
    answers = load_pdf_as_markdown(answers_file) if answers_file.suffix == ".pdf" else load_text_file(answers_file)
    rubric = (
        load_pdf_as_markdown(rubric_file)
        if rubric_file and rubric_file.suffix == ".pdf"
        else load_text_file(rubric_file) if rubric_file
        else None
    )


    # Grade
    if agent_type == "technical":
        result = grade_tech(questions, answers, rubric)
    else:
        result = grade_narrative(questions, answers, rubric)

    print(json.dumps(result, indent=2))


def run_vc(audio_file):
    """Run VC agent with audio input."""
    result = grade_vc(audio_file)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Universal Exam Agent Tester")
    parser.add_argument("--agent", required=True, choices=["technical", "narrative", "vc"],
                        help="Type of grading agent to run")
    parser.add_argument("--questions", type=Path, help="Path to exam questions (PDF or text)")
    parser.add_argument("--answers", type=Path, help="Path to student answers (text)")
    parser.add_argument("--rubric", type=Path, help="Path to rubric (PDF or text)")
    parser.add_argument("--audio", type=Path, help="Path to VC pitch audio file (mp3/wav)")

    args = parser.parse_args()

    if args.agent == "vc":
        if not args.audio:
            raise ValueError("VC agent requires --audio")
        run_vc(args.audio)
    else:
        if not args.questions or not args.answers:
            raise ValueError("Technical/Narrative agents require --questions and --answers")
        run_tech_or_narrative(args.agent, args.questions, args.answers, args.rubric)


if __name__ == "__main__":
    main()
