python run_example.py \
  --agent technical \
  --questions technical_agent/test_input_files/CRA_Final_Examen_Gener_2025_CATALÀ.pdf \
  --answers technical_agent/test_input_files/Respostes_MD.md \
  --rubric technical_agent/test_input_files/CRA_Final_Examen_Rubric.pdf

python run_example.py \
  --agent narrative \
  --questions narrative_agent/test_input_files/exams/exam1.txt \
  --answers narrative_agent/test_input_files/student_answers/exam1_student1.pdf \
  --rubric narrative_agent/test_input_files/exams/rubric.txt

python run_example.py \
  --agent vc \
  --audio vc_pitch_agent/test_input_files/audio/3_Ursify.mp3

