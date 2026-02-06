# YouTube Video Summarizer (Streamlit App)

A Streamlit web application that extracts YouTube video transcripts, chunks them, and generates AI-powered summaries using the FLAN-T5 model. The application provides a user-friendly interface for generating summarized notes and downloading them as a text file.

## Features

- Extracts transcripts from YouTube videos using `youtube-transcript-api`
- Splits long transcripts into manageable chunks
- Uses `google/flan-t5-base` to summarize each chunk
- Combines chunk summaries into a final coherent summary
- Provides a progress bar during summarization
- Allows users to download the final notes as a text file

## Requirements

- Python 3.11 or 3.12
- A virtual environment is recommended

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/youtube-summarizer.git
cd youtube-summarizer
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```
3.Install dependencies:
```bash
pip install -r requirements.txt
```

Running the App

Start the Streamlit app using the following command:
```bash
streamlit run app.py
```

