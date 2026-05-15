# VoxLex-AI-Document-Reader
VoxLex — Upload any document photo and hear it explained in your language. Powered by Gemma 4 vision + offline TTS for 10 languages. Built for the 1 billion people who cannot read.
# VoxLex — Offline Multilingual Document Reader

Hackathon submission for Google Kaggle Gemma 4 Competition

# What it does
VoxLex takes a photo of any document and reads it aloud
in the user's native language. Designed for low-literacy
and non-literate users in underserved communities.

# Supported Languages
Urdu · Hindi · Arabic · Bengali · Persian · 
Indonesian · Swahili · Tamil · Tagalog · Turkish

# Tech Stack
- Vision + LLM: Gemma 4 via Ollama (fully offline)
- TTS: Facebook MMS-TTS (10 languages, offline)
- Backend: Flask + Gunicorn
- Deployment: RunPod GPU / Local CPU

# How to Run
Install dependencies:
pip install -r requirements.txt

Start Ollama and pull model:
ollama pull gemma4:e2b

Run the app:
python app.py
