from sentence_transformers import SentenceTransformer
import numpy as np
import os
import time  # <-- import time ajouté

def load_chunks(file_path="chunks.txt"):
    """Charge les chunks depuis un fichier texte, séparés par '---'."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()
    chunks = [chunk.strip() for chunk in raw.split('---') if chunk.strip()]
    print(f"🔹 {len(chunks)} chunks chargés depuis {file_path}")
    return chunks

def generate_embeddings(chunks, model_name="sentence-transformers/all-mpnet-base-v2", batch_size=32):
    """Génère les embeddings pour une liste de chunks via SentenceTransformer."""
    model = SentenceTransformer(model_name)
    start = time.time()  # début mesure temps
    embeddings = model.encode(chunks, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    end = time.time()  # fin mesure temps
    print(f"⏳ Temps d'encodage : {end - start:.2f} secondes")
    return embeddings

def save_embeddings(embeddings, file_path="embeddings.npy"):
    """Sauvegarde les embeddings au format numpy."""
    np.save(file_path, embeddings)
    print(f"✅ Embeddings sauvegardés dans {file_path}")

def main(chunks_file="chunks.txt", embeddings_file="embeddings.npy", force_regen=False):
    if os.path.exists(embeddings_file):
        if force_regen:
            print(f"⚠️ {embeddings_file} existe déjà, suppression pour régénérer.")
            os.remove(embeddings_file)
        else:
            print(f"ℹ️ {embeddings_file} existe déjà. Utilisez force_regen=True pour régénérer.")
            return

    chunks = load_chunks(chunks_file)
    embeddings = generate_embeddings(chunks)
    save_embeddings(embeddings, embeddings_file)

if __name__ == "__main__":
    # Exemple d'utilisation : 
    # main(force_regen=True)  # pour forcer la régénération
    main(force_regen=True)
