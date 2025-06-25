import os
import json
import pytesseract
import docx
import fitz  # PyMuPDF
from PIL import Image
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import langdetect
import uuid
import spacy
import logging
from transformers import pipeline, Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Load spaCy model ---
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "textcat", "lemmatizer", "attribute_ruler"])
except Exception as e:
    logging.error(f"Failed to load spaCy model: {e}")
    nlp = None

# --- Load transformer summarizer ---
summarizer: Pipeline = None
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
except Exception as e:
    logging.error(f"Failed to load summarizer: {e}")
    summarizer = None

# --- Text Extraction Functions ---
def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file):
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_scanned_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page_index in range(len(doc)):
            pix = doc[page_index].get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img)
    except Exception as e:
        logging.error(f"OCR failed: {e}")
    return text

# --- Helpers ---
def clean_title(text, fallback):
    lines = text.strip().split("\n")[:40]
    candidates = []

    for line in lines:
        line = line.strip()
        if not (10 < len(line) < 120):
            continue
        if any(bad in line.lower() for bad in ["arxiv", "doi", "abstract", "acknowledgements", "copyright"]):
            continue
        if sum(word.lower() in ENGLISH_STOP_WORDS for word in line.split()) >= len(line.split()) / 2:
            continue
        line_clean = line.lstrip("-0123456789. ").strip()
        if line_clean.istitle() and line_clean[0].isupper():
            candidates.append(line_clean)

    if not candidates:
        for line in lines:
            if "mental health" in line.lower() or "introduction" in line.lower():
                candidates.append(line.strip())

    candidates = sorted(candidates, key=lambda x: (-len(x.split()), len(x)))
    return candidates[0] if candidates else fallback

def summarize_text_transformers(text):
    global summarizer
    if summarizer is None:
        try:
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except Exception as e:
            logging.error(f"Runtime load failed: {e}")
            return "Summary not available (summarizer failed to load)"

    try:
        chunks = [" ".join(text.split()[i:i + 500]) for i in range(0, len(text.split()), 500)]
        summary_chunks = []

        for chunk in chunks[:2]:  # Limit to 2 chunks for speed
            if len(chunk.strip().split()) < 50:
                continue
            result = summarizer(chunk, max_length=130, min_length=40, do_sample=False)
            if result and isinstance(result, list) and len(result) > 0:
                summary = result[0].get("summary_text", "")
                if summary:
                    summary_chunks.append(summary)

        final_summary = " ".join(summary_chunks).strip()

        if final_summary:
            return final_summary
        else:
            logging.warning("Empty result from summarizer, using fallback.")
            return " ".join(text.split()[:60]) + "..."

    except Exception as e:
        logging.warning(f"Summarization failed: {e}")
        fallback = " ".join(text.split()[:60]) + "..."
        return fallback if fallback.strip() else "Summary not available"

# --- Metadata Generation ---
def generate_metadata(text, filename, filetype, page_count=None):
    words = text.split()
    logging.info(f"Text length: {len(words)} words")

    # Trim input to 3000 words for speed
    limited_text = " ".join(words[:3000])

    # --- Title extraction
    title = clean_title(limited_text, filename)
    if len(title) > 80:
        title = title[:77] + "..."

    # --- Summary generation
    summary = summarize_text_transformers(limited_text)

    # --- Keyword extraction (TF-based + stopword filtering)
    cleaned_words = [
        word.lower().strip(".,()[]{}\":'")
        for word in words
        if len(word) > 4 and word.lower() not in ENGLISH_STOP_WORDS
    ]
    top_keywords = [word for word, _ in Counter(cleaned_words).most_common(30)]
    keywords = list(dict.fromkeys(top_keywords))[:10]

    # --- Language detection
    try:
        language = langdetect.detect(text[:1000])
    except Exception as e:
        logging.warning(f"Language detection failed: {e}")
        language = "unknown"

    # --- Named entity recognition
    named_entities = []
    if nlp:
        try:
            doc = nlp(limited_text)
            named_entities = list(set(
                ent.text.strip()
                for ent in doc.ents
                if ent.label_ in {"PERSON", "ORG", "GPE", "DATE"} and 2 <= len(ent.text.strip().split()) <= 6
            ))
        except Exception as e:
            logging.warning(f"NER failed: {e}")
    else:
        logging.warning("spaCy NLP model not loaded")

    # --- Construct metadata dict
    metadata = {
        "document_id": str(uuid.uuid4()),
        "title": title,
        "summary": summary,
        "keywords": keywords,
        "file_type": filetype,
        "character_count": len(text),
        "word_count": len(words),
        "uploaded_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "language": language,
        "named_entities": named_entities
    }

    if page_count:
        metadata["page_count"] = page_count

    return metadata
