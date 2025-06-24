import gradio as gr
import json
import tempfile
from pathlib import Path

from technical_agent.tech_grader_agent import grade_exam as grade_tech
from narrative_agent.narrative_grader_agent import grade_exam as grade_narrative
from vc_pitch_agent.vc_grader_agent import grade_pitch as grade_vc
from utils import extract_pdf_to_markdown


def load_text_file(file_path: Path) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def load_file_as_markdown(file_path: Path) -> str:
    if file_path.suffix == ".pdf":
        return extract_pdf_to_markdown(str(file_path))
    return load_text_file(file_path)


def handle_exam(exam_file, rubric_file, response_file, exam_type):
    if not exam_file or not response_file:
        return "Error: Exam and student response are required.", None, None

    questions = load_file_as_markdown(Path(exam_file.name))
    answers = load_file_as_markdown(Path(response_file.name))
    rubric = load_file_as_markdown(Path(rubric_file.name)) if rubric_file else None

    if exam_type == "technical":
        result = grade_tech(questions, answers, rubric)
    else:
        result = grade_narrative(questions, answers, rubric)

    json_output = json.dumps(result, indent=2)

    # Save outputs
    json_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
    with open(json_path, "w") as f:
        f.write(json_output)

    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    # You could implement PDF generation here if needed

    return json_output, json_path, pdf_path


def handle_vc_pitch(audio_file):
    if not audio_file:
        return "Error: Please upload an audio file.", None, None

    result = grade_vc(audio_file)
    json_output = json.dumps(result, indent=2)

    json_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
    with open(json_path, "w") as f:
        f.write(json_output)

    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    # You could implement PDF generation here if needed

    return json_output, json_path, pdf_path


### 🎯 Gradio Interface

exam_tab = gr.Interface(
    fn=handle_exam,
    inputs=[
        gr.File(label="Exam PDF"),
        gr.File(label="Rubric PDF (optional)"),
        gr.File(label="Student Response (.txt, .md, .pdf)"),
        gr.Radio(["narrative", "technical"], label="Exam Type")
    ],
    outputs=[
        gr.Textbox(label="Evaluation Output"),
        gr.File(label="Download JSON"),
        gr.File(label="Download PDF")
    ],
    title="Narrative & Technical Exam Grader"
)

vc_tab = gr.Interface(
    fn=handle_vc_pitch,
    inputs=[gr.Audio(label="Upload VC Pitch (MP3)", type="filepath")],
    outputs=[
        gr.Textbox(label="VC Pitch Evaluation Output"),
        gr.File(label="Download JSON"),
        gr.File(label="Download PDF")
    ],
    title="VC Pitch Grader"
)

gr.TabbedInterface(
    [exam_tab, vc_tab],
    ["Text-based Exams", "VC Pitch Grading"]
).launch()
