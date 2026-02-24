import streamlit as st
import speech_recognition as sr
import nltk
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import tempfile
import os
import string

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

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

    try:
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 3:
            return text
        
        # Remove punctuation and convert to lowercase for word frequency analysis
        words = text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        
        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Calculate word frequencies
        freq_dist = FreqDist(filtered_words)
        
        # Score sentences based on word frequency
        sentence_scores = {}
        for sentence in sentences:
            sentence_words = sentence.lower().translate(str.maketrans('', '', string.punctuation)).split()
            score = sum(freq_dist.get(word, 0) for word in sentence_words if word in filtered_words)
            sentence_scores[sentence] = score
        
        # Get top sentences (approximately 30% of original)
        num_summary_sentences = max(3, len(sentences) // 3)
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_summary_sentences]
        
        # Sort by original order
        top_sentences_ordered = sorted(top_sentences, key=lambda x: sentences.index(x[0]))
        
        summary = ' '.join([sent[0] for sent in top_sentences_ordered])
        return summary
        
    except Exception as e:
        st.warning(f"Summarization failed: {e}. Returning truncated text.")
        return text[:500] + "..." if len(text) > 500 else text


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