# Prerequisite
#1. Install Ollama and the embedding model: mxbai-embed-large, see: https://ollama.com/blog/embedding-models
#2. Install LLM for generating the response, here we will use 'llama3'
#4. Install chromadb for vector embedding database, read more: https://pypi.org/project/chromadb/
#5. Install argparse for handling comand-line arguments, read more: https://www.datacamp.com/tutorial/python-argparse

import json 
import ollama 
import chromadb 
import argparse 

# Load JSON data from file
with open("documents.json", "r") as file:
    documents = json.load(file)

# Initialize ChromaDB Client and Collection
# Creates a ChromaDB client and a new collection named docs.
client = chromadb.Client()
collection = client.create_collection(name="docs") 

# Store each document in a vector embedding database using 'mxbai-embed-large' model from ollama
# if you have not pull the model yet, you need to pull it via terminal: ollama pull mxbai-embed-large
for doc in documents:
    response = ollama.embeddings(model="mxbai-embed-large", prompt=doc['text'])
    embedding = response["embedding"]
    # Adds the document's ID, embedding, and text to the ChromaDB collection.
    collection.add(
        ids=[doc['id']],
        embeddings=[embedding],
        documents=[doc['text']]
    )

# Sets up an argument parser to capture the user's question from the command line.
# Set up argument parser
parser = argparse.ArgumentParser(description='Query ChromaDB')
parser.add_argument('question', type=str, help='The question to ask')

# Parse arguments
args = parser.parse_args()
prompt = args.question

# Generate an embedding for the prompt and retrieve the most relevant document
response = ollama.embeddings(prompt=prompt, model="mxbai-embed-large")
results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
)

retrieved_document = results['documents'][0]
document_id = results['ids'][0]

# Generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
    model="llama3",
    prompt=f"Answer the question based only on the following context:{retrieved_document}. Answer the question only based on the above context: {prompt}. Do not make up the answer if no context can be found from the database."
)

response_text = output['response']
source_info = f"Source: {document_id}"

print("\nResponse:")
print(response_text)
print("\n" + source_info)

# To run: python RAG-business.py "What diploma do I need to be a dental surgeon in France?"