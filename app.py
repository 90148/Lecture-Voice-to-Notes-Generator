import streamlit as st
import speech_recognition as sr
from transformers import pipeline
import nltk
import tempfile
import os

nltk.download('punkt')

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.title("🎤 Lecture Voice-to-Notes Generator")
st.write("Upload your lecture audio file and generate notes, quizzes, and flashcards automatically!")

# Upload Audio
audio_file = st.file_uploader("Upload Lecture Audio (.wav format)", type=["wav"])


def speech_to_text(audio_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language="en-IN")
        return text

    except sr.UnknownValueError:
        return ""

    except sr.RequestError as e:
        st.error(f"Speech Recognition API error: {e}")
        return ""


def summarize_text(text):
    if not text.strip():
        return "⚠️ No clear speech detected to summarize."

    summary = summarizer(
        text,
        max_length=200,
        min_length=50,
        do_sample=False
    )
    return summary[0]['summary_text']


def generate_quiz(summary):
    sentences = summary.split('.')
    quiz = []
    for sentence in sentences[:5]:
        if len(sentence.strip()) > 20:
            quiz.append(f"What is meant by: {sentence.strip()}?")
    return quiz


def generate_flashcards(summary):
    sentences = summary.split('.')
    flashcards = []
    for sentence in sentences[:5]:
        if len(sentence.strip()) > 20:
            flashcards.append({
                "front": sentence.strip(),
                "back": f"Explanation of: {sentence.strip()}"
            })
    return flashcards


if audio_file is not None:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    try:
        st.info("🎧 Converting speech to text...")
        transcript = speech_to_text(tmp_path)

        if not transcript:
            st.error("❌ Could not understand the audio. Please upload a clearer WAV file.")
            st.stop()

        st.subheader("📜 Transcript")
        st.write(transcript)

        st.info("🧠 Generating Summary...")
        summary = summarize_text(transcript)

        st.subheader("📝 Study Notes")
        st.write(summary)

        st.subheader("❓ Quiz Questions")
        quiz = generate_quiz(summary)
        for i, q in enumerate(quiz, 1):
            st.write(f"{i}. {q}")

        st.subheader("📚 Flashcards")
        flashcards = generate_flashcards(summary)
        for card in flashcards:
            st.write(f"**Front:** {card['front']}")
            st.write(f"**Back:** {card['back']}")
            st.write("---")

        # Download Notes
        notes_text = f"""
Transcript:
{transcript}

Summary:
{summary}

Quiz:
{quiz}

Flashcards:
{flashcards}
"""

        st.download_button(
            label="📥 Download Notes",
            data=notes_text,
            file_name="lecture_notes.txt",
            mime="text/plain"
        )

    finally:
        os.remove(tmp_path)