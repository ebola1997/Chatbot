import ollama
import os
import json
import numpy as np
from numpy.linalg import norm
import PyPDF2
from docx import Document
import pandas as pd
import streamlit as st

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

# Membaca file CSV
# Membaca file CSV tanpa header dan mengisi NaN
def read_csv(filename):
    df = pd.read_csv(filename, header=None, delimiter=';', skip_blank_lines=True)
    df = df.fillna("0")  # Replace NaN with "0"
    text = ""
    
    # Loop through the rows
    for idx, row in df.iterrows():
        if idx >= 1:  # Ensure that idx is an integer
            text += f"'{idx}. " + ", ".join(str(value) for value in row.values) + "\n"
        else:
            # If it's the first row (header row), don't add an index number
            text += "Ini adalah data csv dengan delimter (,)\n" + ", ".join(str(value) for value in row.values) + "\n"
    
    return text

# Membaca file Excel (.xlsx)
def read_xlsx(filename):
    df = pd.read_excel(filename)
    text = ""
    for index, row in df.iterrows():
        text += " ".join(str(value) for value in row.values) + "\n"
    return text

# Mendapatkan paragraf dari semua file
def parse_file(filename):
    if filename.endswith(".txt"):
        content = read_txt(filename)
    elif filename.endswith(".pdf"):
        content = read_pdf(filename)
    elif filename.endswith(".docx"):
        content = read_docx(filename)
    elif filename.endswith(".csv"):
        content = read_csv(filename)
    elif filename.endswith(".xlsx"):  # Menambahkan pengecekan untuk file Excel
        content = read_xlsx(filename)
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

# Streamlit App
def main():
    SYSTEM_PROMPT = """You are an assistant that answers questions only in Bahasa Indonesia. 
    Your answers must be based solely on the provided context extracted from the documents. 
    If the answer cannot be determined from the context, respond with "Maaf, saya tidak tahu." 
    Do not include any information outside of the given context, and strictly reply in Bahasa Indonesia.

    Context:
    """

    # Load subfolders in the `data` directory
    data_root = "data/"
    subfolders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]

    # Streamlit select box to choose the folder
    selected_folder = st.selectbox("Pilih folder dataset:", subfolders)

    # Load or process embeddings
    data_folder = os.path.join(data_root, selected_folder)
    all_paragraphs = []
    filenames = []

    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        if file.lower().endswith((".txt", ".pdf", ".docx", ".csv")):
            paragraphs = parse_file(file_path)
            all_paragraphs.extend(paragraphs)
            filenames.append(file)


    folder_name = os.path.basename(data_folder)

    # Buat nama file embeddings berdasarkan nama folder
    embeddings_filename = f"data_embeddings_{folder_name}"
    # Buat embedding
    embeddings = get_embeddings(embeddings_filename, "nomic-embed-text", all_paragraphs)

    # Streamlit UI
    st.title("Chatbot dengan RAG (Retrieval-Augmented Generation)")
    st.write(f"Ajukan pertanyaan berdasarkan data yang ada di folder `{selected_folder}`.")

    # Menyimpan riwayat percakapan
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input pengguna
    user_input = st.text_input("Pertanyaan Anda:", key="input")
    if st.button("Kirim") and user_input.strip():
        # Proses pertanyaan
        prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=user_input)["embedding"]
        most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

        # Generate response
        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                    + "\n".join(all_paragraphs[item[1]] for item in most_similar_chunks),
                },
                {"role": "user", "content": user_input},
            ],
        )
        response_text = response["message"]["content"]

        # Simpan ke riwayat
        st.session_state.chat_history.append({"user": user_input, "bot": response_text})

    # Tampilkan riwayat percakapan
    st.subheader("Riwayat Percakapan")
    for chat in st.session_state.chat_history:
        st.markdown(f"**Anda:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

if __name__ == "__main__":
    main()

