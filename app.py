import streamlit as st
import PyPDF2
import docx
import nltk
from nltk.corpus import stopwords
from collections import Counter
import spacy

# Download NLTK data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Please download the spaCy model by running: python -m spacy download en_core_web_sm")
    st.stop()

# Text and metadata extraction functions
def extract_text_and_metadata_pdf(file):
    file.seek(0)
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    metadata = pdf.metadata
    creation_date = metadata.get('/CreationDate', None)
    modification_date = metadata.get('/ModDate', None)
    return text, creation_date, modification_date

def extract_text_and_metadata_docx(file):
    file.seek(0)
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    core_properties = doc.core_properties
    creation_date = core_properties.created
    modification_date = core_properties.modified
    return text, creation_date, modification_date

def extract_text_txt(file):
    file.seek(0)
    text = file.read().decode("utf-8")
    return text, None, None

# Semantic analysis functions
def extract_keywords(text, num_keywords=5):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    word_freq = Counter(words)
    return [word for word, _ in word_freq.most_common(num_keywords)]

def extract_entities(text):
    doc = nlp(text[:10000])  # Limit to first 10,000 characters
    entities = {
        'persons': [ent.text for ent in doc.ents if ent.label_ == 'PERSON'],
        'organizations': [ent.text for ent in doc.ents if ent.label_ == 'ORG'],
        'dates': [ent.text for ent in doc.ents if ent.label_ == 'DATE'],
        'locations': [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    }
    return entities

# Streamlit app
st.title("Automated Metadata Generator")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    file_name = uploaded_file.name
    file_size = uploaded_file.size

    # Extract text and metadata based on file type
    if file_type == "application/pdf":
        text, creation_date, modification_date = extract_text_and_metadata_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text, creation_date, modification_date = extract_text_and_metadata_docx(uploaded_file)
    elif file_type == "text/plain":
        text, creation_date, modification_date = extract_text_txt(uploaded_file)
    else:
        st.error("Unsupported file format")
        st.stop()

    # Generate metadata
    keywords = extract_keywords(text)
    entities = extract_entities(text)
    word_count = len(text.split())
    character_count = len(text)

    metadata = {
        "filename": file_name,
        "file_type": file_type,
        "size": file_size,
        "creation_date": str(creation_date) if creation_date else None,
        "modification_date": str(modification_date) if modification_date else None,
        "keywords": keywords,
        "entities": entities,
        "word_count": word_count,
        "character_count": character_count
    }

    st.write("Metadata:")
    st.json(metadata)
