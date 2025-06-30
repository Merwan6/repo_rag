import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity

# 1. Charger les chunks
with open("chunks.txt", "r", encoding="utf-8") as f:
    raw_chunks = f.read().split("\n---\n")
chunks = [c.strip() for c in raw_chunks if c.strip()]

# 2. Charger les embeddings
embeddings = np.load("embeddings.npy")

# 3. Charger le modèle d'embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# 4. Charger le modèle génératif et tokenizer
gen_model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen_model = gen_model.to(device)

SIMILARITY_THRESHOLD = 0.4  # seuil minimum pour considérer un chunk pertinent

def extract_title_from_chunk(chunk):
    # Extrait le titre en début de chunk selon format "ARTICLE X - Titre"
    match = re.search(r'(ARTICLE\s+\d+\s*(?:bis|ter)?\s*-\s*[^\n]+)', chunk, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # Si pas trouvé, retourne un extrait
        return chunk[:50].strip() + "..."

def get_first_sentence(text):
    # Nettoyer numéros en début de ligne (ex: "1.", "2.", etc.)
    text = re.sub(r'^\s*\d+(\.\d+)*\s*\.\s*', '', text, flags=re.MULTILINE)
    # Remplacer les retours à la ligne par espace
    text = re.sub(r'\s*\n\s*', ' ', text)

    # Chercher toutes les phrases (finies par ., ! ou ?)
    sentences = re.findall(r'.+?[.!?](?=\s+|$)', text)

    if sentences:
        first_sentence = sentences[0].strip()

        # Si trop courte (ex: "I."), concaténer la deuxième phrase si possible
        if len(first_sentence) < 20 and len(sentences) > 1:
            combined = first_sentence + " " + sentences[1].strip()
            return combined
        else:
            return first_sentence
    else:
        # Pas de phrase détectée, retourner tout le texte
        return text.strip()


def retrieve_top_chunks(question, top_k=3):
    question_emb = embedding_model.encode([question])
    similarities = cosine_similarity(question_emb, embeddings)[0]
    valid_indices = [i for i, score in enumerate(similarities) if score >= SIMILARITY_THRESHOLD]
    if not valid_indices:
        return [], [], []
    valid_indices.sort(key=lambda i: similarities[i], reverse=True)
    selected_indices = valid_indices[:top_k]
    selected_chunks = [chunks[i] for i in selected_indices]
    selected_scores = [similarities[i] for i in selected_indices]
    return selected_chunks, selected_scores, selected_indices

def truncate_context(context, max_tokens=400):
    tokens = tokenizer.tokenize(context)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)

def generate_answer(question, context):
    truncated_context = truncate_context(context, max_tokens=400)
    input_text = (
        "Tu es un assistant expert en règlements sportifs, précis et naturel.\n"
        f"Voici un extrait de document officiel :\n{truncated_context}\n\n"
        "Réponds en une phrase claire, sans recopier le texte mot à mot, à la question suivante :\n"
        f"👉 {question}\n"
        "Si l’information n’est pas présente, réponds simplement : \"Je ne sais pas.\""
    )
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = gen_model.generate(
        **inputs,
        max_length=80,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def rag_qa(question):
    relevant_chunks, scores, indices = retrieve_top_chunks(question, top_k=3)
    if not relevant_chunks:
        return "Je ne sais pas.", ""

    # Premier article complet avec score juste après le titre (même ligne)
    first_chunk = chunks[indices[0]].strip()
    # Extraire le titre du 1er article
    title_match = re.search(r'(ARTICLE\s+\d+\s*(?:bis|ter)?\s*-\s*[^\n]+)', first_chunk, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).upper()
        # Remplacer la première ligne du chunk (le titre) par titre + score
        first_chunk_lines = first_chunk.split('\n')
        first_chunk_lines[0] = f"{title} (score: {scores[0]:.3f})"
        full_first_article_with_score = "\n".join(first_chunk_lines)
    else:
        full_first_article_with_score = f"(score: {scores[0]:.3f})\n{first_chunk}"

    # Construire la liste des articles 2 et 3 au format souhaité
    articles_summary = ""
    for i, (idx, score) in enumerate(zip(indices, scores)):
        if i == 0:
            continue
        chunk_text = chunks[idx].strip()
        title_match = re.search(r'(ARTICLE\s+\d+\s*(?:bis|ter)?\s*-\s*[^\n]+)', chunk_text, re.IGNORECASE)
        if title_match:
            title = title_match.group(1).upper()
            # Supprimer le titre du chunk pour éviter redondance dans la phrase
            chunk_text_clean = chunk_text.replace(title_match.group(0), '').strip()
        else:
            title = "ARTICLE INCONNU"
            chunk_text_clean = chunk_text

        first_sentence = get_first_sentence(chunk_text_clean)
        articles_summary += f"{i+1}. {title} (score: {score:.3f})\n{first_sentence}\n\n"

    return full_first_article_with_score, articles_summary


# Interface Gradio modifiée avec un seul output
iface = gr.Interface(
    fn=rag_qa,
    inputs=gr.Textbox(lines=2, placeholder="Pose ta question ici..."),
    outputs=[
        gr.Textbox(label="Réponse générée"),
        gr.Textbox(label="Articles les plus probables")
    ],
    title="Chat avec ton PDF - RAG System",
    description="Pose une question, je réponds avec le contenu du PDF."
)

if __name__ == "__main__":
    iface.launch()