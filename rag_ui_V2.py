import ollama
import os
import json
import numpy as np
from numpy.linalg import norm
import PyPDF2
from docx import Document
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

# Streamlit App
def main():
    SYSTEM_PROMPT = """Anda adalah asisten yang membantu menjawab pertanyaan dengan bahasa Indonesia 
    dan berdasarkan cuplikan teks yang diberikan dalam konteks. Jawab hanya menggunakan konteks yang disediakan, 
    menjadi sesingkat mungkin. Jika Anda tidak yakin, katakan saja Anda tidak tahu.
    Context:
    """

    # Load subfolders in the `data` directory
    data_root = "data"
    subfolders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]

    # Streamlit select box to choose the folder
    selected_folder = st.selectbox("Pilih folder dataset:", subfolders)

    # Load or process embeddings
    data_folder = os.path.join(data_root, selected_folder)
    all_paragraphs = []
    filenames = []

    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        if file.lower().endswith((".txt", ".pdf", ".docx")):
            paragraphs = parse_file(file_path)
            all_paragraphs.extend(paragraphs)
            filenames.append(file)

    embeddings = get_embeddings(f"{selected_folder}_embeddings", "bangundwir/bahasa-4b-v2:latest", all_paragraphs)

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
        prompt_embedding = ollama.embeddings(model="bangundwir/bahasa-4b-v2:latest", prompt=user_input)["embedding"]
        most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

        # Generate response
        response = ollama.chat(
            model="bangundwir/bahasa-4b-v2:latest",
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
