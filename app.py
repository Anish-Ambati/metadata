
import streamlit as st
import os
import json
import tempfile
import PyPDF2
import docx
import pytesseract
from PIL import Image, ImageDraw
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Load the spaCy English model from local path
# Ensure the 'spacy_models/en_core_web_sm' directory
# containing the model files is in your repository.
try:
    # Use the relative path to the model directory within your repo
    model_path = "./spacy_models/en_core_web_sm"
    if not os.path.exists(model_path):
         st.error(f"Error: spaCy model not found at {model_path}. Make sure the 'spacy_models/en_core_web_sm' directory is in your repository.")
         st.stop() # Stop the Streamlit app if model is not found
    nlp = spacy.load(model_path)
    st.success("spaCy model loaded successfully!")
except Exception as e:
    st.error(f"Error loading spaCy model: {e}")
    st.stop() # Stop the Streamlit app if loading fails


# Document Extraction Functions (from previous steps)
def extract_text_from_pdf(pdf_path):
    \"\"\"Extracts text content from a PDF file.\"\"\"
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or "" # Handle potential None return from extract_text
    except FileNotFoundError:
        return f"Error: PDF file not found at {{pdf_path}}"
    except Exception as e:
        return f"Error extracting text from PDF: {{e}}"
    return text

def extract_text_from_docx(docx_path):
    \"\"\"Extracts text content from a DOCX file.\"\"\"
    text = ""
    try:
        doc = docx.Document(docx_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\\n" # Use double backslash for newline in string literal
    except FileNotFoundError:
        return f"Error: DOCX file not found at {{docx_path}}"
    except Exception as e:
        return f"Error extracting text from DOCX: {{e}}"
    return text

def extract_text_from_txt(txt_path):
    \"\"\"Extracts text content from a TXT file.\"\"\"
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        return f"Error: TXT file not found at {{txt_path}}"
    except Exception as e:
        return f"Error extracting text from TXT: {{e}}"
    return text

def perform_ocr_on_image(image_input):
    \"\"\"Performs OCR on an image (file path or PIL Image object).\"\"\"
    try:
        if isinstance(image_input, str):
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            return "Error: Invalid input for OCR. Must be file path or PIL Image."

        text = pytesseract.image_to_string(img)
        return text
    except FileNotFoundError:
        return f"Error: Image file not found for OCR at {{image_input}}"
    except Exception as e:
        return f"Error performing OCR: {{e}}"


def extract_document_text(file_path):
    \"\"\"Determines file type and extracts text using appropriate function.\"\"\"
    if not os.path.exists(file_path):
        return f"Error: File not found at {{file_path}}"

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        # Basic PDF text extraction first
        text = extract_text_from_pdf(file_path)
        # TODO: Implement logic to check if PDF is scanned and needs OCR
        # This would likely involve analyzing image content within the PDF
        # For now, we'll just return the extracted text.
        return text
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    # TODO: Add handling for image files if needed as per instruction 5
    # elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff']:
    #     return perform_ocr_on_image(file_path)
    else:
        return f"Error: Unsupported file type: {{file_extension}}"

# Semantic Content Identification Functions (from previous steps)
def extract_entities(text):
    \"\"\"Extracts named entities from text using spaCy.\"\"\"
    if not text:
        return []
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_keywords(text):
    \"\"\"Extracts keywords from text based on noun chunks and named entities.\"\"\"
    if not text:
        return []
    doc = nlp(text)
    keywords = [chunk.text for chunk in doc.noun_chunks] + [ent.text for ent in doc.ents]
    return list(set(keywords)) # Remove duplicates

def perform_topic_modeling(text, num_topics=3, num_words=5):
    \"\"\"Performs basic topic modeling using TF-IDF and NMF.\"\"\"
    if not text or len(text.split()) < 10: # Basic check for sufficient text
        return "Insufficient text for topic modeling."

    try:
        vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, stop_words='english')
        tfidf = vectorizer.fit_transform([text])

        # Ensure n_components is less than or equal to n_samples and n_features
        n_components = min(num_topics, tfidf.shape[0], tfidf.shape[1])
        if n_components <= 0:
             return "Could not determine topics (insufficient features)."


        nmf_model = NMF(n_components=n_components, random_state=1, init='nndsvd')
        nmf_model.fit(tfidf)

        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(nmf_model.components_):
            top_words_indices = topic.argsort()[:-num_words - 1:-1]
            top_words = [feature_names[i] for i in top_words_indices]
            topics[f"Topic {{topic_idx+1}}"] = top_words
        return topics

    except Exception as e:
        return f"Error during topic modeling: {{e}}"

def analyze_document_semantics(text):
    """
    Performs semantic analysis on the extracted text.

    Args:
        text: The extracted text content of the document.

    Returns:
        A dictionary containing extracted entities, keywords, and topics.
        Returns an error message if analysis fails.
    """
    if not text:
        return {{"error": "No text provided for semantic analysis."}}

    results = {{}}
    try:
        results["entities"] = extract_entities(text)
    except Exception as e:
        results["entities"] = f"Error extracting entities: {{e}}"

    try:
        results["keywords"] = extract_keywords(text)
    except Exception as e:
        results["keywords"] = f"Error extracting keywords: {{e}}"

    try:
        topic_modeling_result = perform_topic_modeling(text)
        if isinstance(topic_modeling_result, str) and "Insufficient text" in topic_modeling_result:
             results["topics"] = topic_modeling_result
        elif isinstance(topic_modeling_result, str) and "Error" in topic_modeling_result:
             results["topics"] = topic_modeling_result
        else:
             results["topics"] = topic_modeling_result

    except Exception as e:
        results["topics"] = f"Error performing topic modeling: {{e}}"

    return results

# Metadata Generation Function (from previous steps)
def generate_metadata(semantic_results):
    """
    Generates structured metadata from semantic analysis results.

    Args:
        semantic_results: A dictionary containing entities, keywords, and topics
                          from the analyze_document_semantics function.

    Returns:
        A dictionary containing structured metadata.
    """
    metadata = {{
        'entities': semantic_results.get('entities', []),
        'keywords': semantic_results.get('keywords', []),
        'topics': semantic_results.get('topics', "Topic modeling results not available."),
        'summary': "Summary generation not yet implemented.",
        'categories': "Categorization not yet implemented."
    }}
    return metadata


# Streamlit App Logic
st.title("Automated Document Metadata Generation")

uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'docx', 'txt'])

st.info("Supported file types: PDF, DOCX, TXT")

if st.button("Generate Metadata"):
    if uploaded_file is not None:
        st.write("Processing document...")

        # Save the uploaded file temporarily
        try:
            # Create a temporary directory and file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Call the extraction function
            text = extract_document_text(tmp_path)

            if "Error" in text:
                st.error(f"Error extracting text: {text}")
            else:
                st.write("Text extracted successfully. Analyzing semantics...")
                # Call the semantic analysis function
                semantic_results = analyze_document_semantics(text)

                if "error" in semantic_results:
                    st.error(f"Error during semantic analysis: {semantic_results['error']}")
                else:
                    st.write("Semantic analysis complete. Generating metadata...")
                    # Call the metadata generation function
                    metadata = generate_metadata(semantic_results)

                    st.write("Generated metadata:")
                    st.json(metadata)

        except Exception as e:
            st.error(f"An unexpected error occurred during processing: {e}")
        finally:
            # Clean up the temporary file
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    else:
        st.warning("Please upload a file first.")

