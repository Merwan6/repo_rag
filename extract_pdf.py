import pdfplumber

def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text

if __name__ == "__main__":
    pdf_file = "reglement-des-championnats-nationaux-de-jeunes-2024-2025.pdf"
    text = extract_text_from_pdf(pdf_file)
    
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    print("✅ Texte extrait et sauvegardé dans extracted_text.txt")
