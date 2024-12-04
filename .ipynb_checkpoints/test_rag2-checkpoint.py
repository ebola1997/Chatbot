import ollama
import time
import os
import json
import numpy as np
from numpy.linalg import norm
import PyPDF2
from docx import Document

# Membaca file TXT
def read_txt(filename):
    with open(filename, "r", encoding="utf-8-sig") as f:
        return f.read()

# Membaca file PDF
def read_pdf(filename):
    with open(filename, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

# Membaca file DOCX
def read_docx(filename):
    doc = Document(filename)
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    return text

# Mendapatkan paragraf dari semua file
def parse_file(filename):
    if filename.endswith(".txt"):
        content = read_txt(filename)
    elif filename.endswith(".pdf"):
        content = read_pdf(filename)
    elif filename.endswith(".docx"):
        content = read_docx(filename)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    
    paragraphs = []
    buffer = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            buffer.append(line)
        elif len(buffer):
            paragraphs.append(" ".join(buffer))
            buffer = []
    if len(buffer):
        paragraphs.append(" ".join(buffer))
    return paragraphs

# Simpan embedding ke file
def save_embeddings(filename, embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

# Memuat embedding dari file
def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)

# Mendapatkan embedding
def get_embeddings(filename, modelname, chunks):
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    save_embeddings(filename, embeddings)
    return embeddings

# Cosine similarity untuk menemukan kemiripan
def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

# Fungsi utama
def main():
    SYSTEM_PROMPT = """Anda adalah asisten yang membantu menjawab pertanyaan dengan bahasa Indonesia 
    dan berdasarkan cuplikan teks yang diberikan dalam konteks. Jawab hanya menggunakan konteks yang disediakan, 
    menjadi sesingkat mungkin. Jika Anda tidak yakin, katakan saja Anda tidak tahu.
    Context:
    """

    data_folder = "data"
    all_paragraphs = []
    filenames = []

    # Iterasi semua file dalam folder data
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        if file.lower().endswith((".txt", ".pdf", ".docx")):
            paragraphs = parse_file(file_path)
            all_paragraphs.extend(paragraphs)
            filenames.append(file)

    # Buat embedding
    embeddings = get_embeddings("data_embeddings", "nomic-embed-text", all_paragraphs)

    while True:
        prompt = input("Silakan tanya bosku? (ketik 'exit' untuk keluar) -> ")
        
        if prompt.lower() == "exit":
            print("Exiting the assistant. Goodbye!")
            break

        prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]
        most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                    + "\n".join(all_paragraphs[item[1]] for item in most_similar_chunks),
                },
                {"role": "user", "content": prompt},
            ],
        )
        
        print("\n\n")
        print(response["message"]["content"])


if __name__ == "__main__":
    main()