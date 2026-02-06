import re
import torch
import streamlit as st
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------
# Helper Functions
# -----------------------------

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


def get_transcript(video_id):
    """Fetch transcript using the NEW API format."""
    try:
        api = YouTubeTranscriptApi()
        # The .fetch method grabs the subtitle object list
        transcript = api.fetch(video_id)
        # We join the list into a single long string of text
        return " ".join([t.text for t in transcript])

    except TranscriptsDisabled:
        return "Error: Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "Error: No transcript found for this video."
    except Exception as e:
        return f"Error: {str(e)}"




def chunk_text(text, chunk_size=1200):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# -----------------------------
# Model Loading (Cached)
# -----------------------------

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    return tokenizer, model, device


def summarize_chunk(text_chunk, tokenizer, model, device):
    prompt = f"Summarize the following text clearly:\n{text_chunk}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    summary_ids = model.generate(
        **inputs,
        max_new_tokens=120,
        num_beams=4,
        length_penalty=1.0,
        early_stopping=True,
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(
    page_title="YouTube AI Notes Generator",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 YouTube AI Notes Generator")
st.write("Paste a YouTube link and get clean, structured AI-generated notes.")

url = st.text_input("📺 YouTube Video URL")

chunk_size = st.slider(
    "Transcript chunk size",
    min_value=600,
    max_value=2000,
    value=1200,
    step=100,
)

if st.button("🧠 Generate Notes") and url:
    video_id = extract_video_id(url)

    if not video_id:
        st.error("Invalid YouTube URL.")
        st.stop()

    with st.status("🎧 Fetching transcript...", expanded=False):
        transcript = get_transcript(video_id)

    if transcript.startswith("Error"):
        st.error(transcript)
        st.stop()

    chunks = chunk_text(transcript, chunk_size=chunk_size)
    st.info(f"🔪 Transcript split into **{len(chunks)} chunks**")

    tokenizer, model, device = load_model()

    notes = []
    progress = st.progress(0)

    st.subheader("📝 AI Generated Notes")

    for i, chunk in enumerate(chunks):
        summary = summarize_chunk(chunk, tokenizer, model, device)
        notes.append(f"- {summary}")
        progress.progress((i + 1) / len(chunks))

    st.markdown("\n".join(notes))

    st.download_button(
        label="⬇️ Download Notes",
        data="\n".join(notes),
        file_name="youtube_ai_notes.txt",
        mime="text/plain",
    )

elif not url:
    st.caption("👆 Enter a YouTube URL to get started")
