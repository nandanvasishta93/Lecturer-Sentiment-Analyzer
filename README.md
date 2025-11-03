
# 🎤 Lecturer Sentiment Analyzer

An AI-powered web application that helps educators improve their teaching delivery through automated sentiment analysis and speech analytics.

## Features
- 📊 Sentiment analysis of lecture audio
- 🎤 Live recording and file upload support
- 📈 Speaking metrics (pace, filler words, vocabulary)
- 💡 Personalized feedback and suggestions
- 🌐 Web-based dashboard

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nandanvasishta93/lecturer-sentiment-analyzer.git
cd lecturer-sentiment-analyzer
Install dependencies:

bash
pip install -r requirements.txt
Run the application:

bash
python app.py
Open http://localhost:5000 in your browser

Technologies Used
Frontend: HTML, CSS, JavaScript, Bootstrap

Backend: Flask, Python

Audio Processing: SpeechRecognition, PyAudio

NLP: TextBlob, NLTK

Usage
Upload audio files (WAV, MP3, OGG, M4A, FLAC)

View sentiment analysis and speaking metrics

Receive actionable feedback for improvement

text

### **`run.py`** (Optional)
```python
#!/usr/bin/env python3
"""
Entry point for Lecturer Sentiment Analyzer
"""
from app import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)