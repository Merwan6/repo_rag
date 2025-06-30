import json
import ollama
import chromadb
import argparse

def load_documents(filepath):
    with open(filepath, "r") as file:
        return json.load(file)

def store_documents(collection, documents):
    for doc in documents:
        response = ollama.embeddings(model="mxbai-embed-large", prompt=doc['text'])
        embedding = response["embedding"]
        collection.add(
            ids=[doc['id']],
            embeddings=[embedding],
            documents=[doc['text']]
        )

def main(question):
    # Load JSON data from file
    documents = load_documents("documents.json")

    client = chromadb.Client()
    collection = client.create_collection(name="docs")

    # Store documents in the collection
    store_documents(collection, documents)

    # Generate an embedding for the prompt and retrieve the top 3 most relevant documents
    response = ollama.embeddings(prompt=question, model="mxbai-embed-large")
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=3
    )

    retrieved_documents = results['documents'][0]
    document_ids = results['ids'][0]

    # Prepare the context by concatenating the top 3 documents
    context = "\n".join(retrieved_documents)
    source_info = "\n".join([f"Source: {doc_id}" for doc_id in document_ids])

    # Generate a response combining the prompt and data we retrieved
    output = ollama.generate(
        model="llama3",
        prompt=f"Using this data:\n{context}\nRespond to this prompt: {question}"
    )

    response_text = output['response']

    print("\nResponse:")
    print(response_text)
    print("\n" + source_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query ChromaDB')
    parser.add_argument('question', type=str, help='The question to ask')
    args = parser.parse_args()

    main(args.question)
