import pdfplumber # For extracting text from PDF files.
import os # For interacting with the operating system, such as reading files from directories.
import json # For handling JSON data.
import spacy # For natural language processing tasks.

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Extracts text from each page and concatenates it into a single string.
def pdf_to_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n\n"  # Ensuring paragraph breaks are preserved
    return text

# Processes text using spaCy to identify sentence boundaries.
# Chunks the text into smaller parts, ensuring each chunk does not exceed a specified size (1000 characters in this case).
def chunk_text(text, nlp):
    doc = nlp(text)
    chunks = []
    current_chunk = ""
    for sent in doc.sents:
        if len(current_chunk) + len(sent.text) > 1000:  # Adjust chunk size based on needs
            chunks.append(current_chunk.strip())
            current_chunk = sent.text
        else:
            current_chunk += " " + sent.text
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# Iterates through all PDF files in a specified directory.
# Converts each PDF to text and then chunks the text.
# Creates a list of documents with unique IDs and text chunks.
def process_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            path = os.path.join(directory, filename)
            text = pdf_to_text(path)
            chunks = chunk_text(text, nlp)
            for index, chunk in enumerate(chunks):
                doc_id = f"{filename}:part{index+1}:{index+1}"
                documents.append({
                    'id': doc_id,
                    'text': chunk
                })
    return documents

# Implementation:
documents = process_directory('data_business') # name of the folder of the data
json.dump(documents, open('documents.json', 'w'))
