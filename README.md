# 🧠 Chat PDF avec Retrieval-Augmented Generation (RAG)

Ce projet implémente un système de **Retrieval-Augmented Generation (RAG)** permettant à un utilisateur de poser des questions en langage naturel et d’obtenir des réponses **contextualisées** basées sur le contenu d’un **document PDF** (ex : règlement sportif, document juridique, etc.).

## 🚀 Fonctionnalités

- 🔍 Extraction de texte depuis un fichier PDF (`pdfplumber`)
- ✂️ Segmentation du texte en articles par expressions régulières
- 🧠 Embeddings générés avec `sentence-transformers/all-mpnet-base-v2`
- 📌 Recherche vectorielle par similarité cosinus
- ✍️ Génération de réponse contextuelle avec `google/flan-t5-large`
- 💬 Interface interactive via **Gradio**
- 📄 Affichage des articles utilisés pour la réponse

---

## 📁 Structure du projet

```yaml
repo_rag/
│
├── extract_text.py          # Extraction du texte depuis le PDF
├── split_text.py            # Segmentation en articles
├── generate_embeddings.py   # Génération des embeddings
├── rag_chat.py              # Interface utilisateur avec Gradio
│
├── extracted_text.txt       # Texte brut extrait du PDF
├── chunks.txt               # Articles découpés et nettoyés
├── embeddings.npy           # Embeddings sauvegardés
├── requirements.txt         # Liste des dépendances Python
└── README.md                # Présentation du projet
```

---

## 🛠️ Installation

### Prérequis

- Python 3.8+
- `pip` ou `conda`

### Étapes

1. **Cloner le dépôt**

```bash
git clone https://github.com/Merwan6/repo_rag.git
cd repo_rag
```

2. **Créer un environnement virtuel** (optionnel mais recommandé)

```bash
python -m venv venv
source venv/bin/activate  # sous Linux/macOS
venv\Scripts\activate     # sous Windows
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Lancer l’application**

```bash
python rag_chat.py
```
