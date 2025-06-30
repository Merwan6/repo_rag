import re

def split_text_by_articles(text):
    # Séparer préambule s’il y en a : tout ce qui précède le premier "ARTICLE ..."
    article_pattern = re.compile(r'ARTICLE\s+\d+\s*(?:bis|ter)?\s*-\s*[^\n]+', re.IGNORECASE)

    # Trouver le premier article
    first_article_match = article_pattern.search(text)
    articles = []

    if first_article_match:
        preambule = text[:first_article_match.start()].strip()
        if preambule:
            articles.append(("PREAMBULE", preambule))

        # Trouver tous les articles (début)
        matches = list(article_pattern.finditer(text))

        for i in range(len(matches)):
            start_idx = matches[i].start()
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
            chunk = text[start_idx:end_idx].strip()

            # Extraire le titre de l'article (ex: "ARTICLE 2 - MODALITES...")
            title_match = re.match(r'(ARTICLE\s+\d+\s*(?:bis|ter)?\s*-\s*[^\n]+)', chunk, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else f"ARTICLE {i+1}"

            articles.append((title, chunk))
    else:
        # Pas d'article détecté, on renvoie tout comme un seul chunk
        articles.append(("TEXTE COMPLET", text.strip()))

    print(f"🔹 Nombre total d'articles détectés (et préambule) : {len(articles)}")
    return articles

if __name__ == "__main__":
    with open("extracted_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    articles = split_text_by_articles(text)

    with open("chunks.txt", "w", encoding="utf-8") as f:
        for title, content in articles:
            f.write(f"{title}\n{content}\n---\n")

    print(f"✅ {len(articles)} articles (et préambule) sauvegardés dans chunks.txt")
