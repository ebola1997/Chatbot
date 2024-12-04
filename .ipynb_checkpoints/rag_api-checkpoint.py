import ollama
import os
import json
import numpy as np
from numpy.linalg import norm
import PyPDF2
from docx import Document
import streamlit as st
import requests

# URL untuk API Ollama, ganti sesuai dengan alamat API Anda
OLLAMA_API_URL = "http://localhost:11434/api/"

# Fungsi untuk request embeddings
def get_embeddings_from_ollama(modelname, chunk):
    response = requests.post(
        OLLAMA_API_URL + "embeddings",
        json={"model": modelname, "prompt": chunk},
    )
    response.raise_for_status()
    return response.json()["embedding"]

# Fungsi untuk request chat (LLM) response
import requests
import json

# Fungsi untuk request chat (LLM) response
def chat_with_ollama(model, messages):
    try:
        response = requests.post(
            OLLAMA_API_URL + "chat",
            json={"model": model, "messages": messages},
        )
        
        if response.headers.get('Content-Type') == 'application/x-ndjson':
            # print("Response is in NDJSON format.")
            
            full_response = ""  # Initialize an empty string to hold the full response
            
            for line in response.text.splitlines():
                try:
                    # Parse each line as a JSON object
                    json_data = json.loads(line)
                    
                    # Print the entire json_data to inspect its structure
                    # print("Parsed JSON Data:", json_data)
                    
                    # Check if 'message' key exists and contains 'content'
                    if 'message' in json_data and 'content' in json_data['message']:
                        # Append the 'content' part of each 'message' to the full response
                        full_response += json_data['message']['content']
                        
                        # If the response is complete, break out of the loop
                        if json_data.get('done'):
                            break
                    else:
                        print("No 'message' or 'content' key in this JSON object:", json_data)
                
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line}")
                    print(f"JSONDecodeError: {e}")
        
            # Return the full response after the loop ends
            return full_response

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None



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

# Mendapatkan embedding
def get_embeddings(filename, modelname, chunks):
    embedding_file_path = f"embeddings/{filename}.json"
    
    # Check if the embedding file already exists
    if os.path.exists(embedding_file_path):
        print(f"Embedding file {filename}.json already exists. Loading embeddings...")
        # Load the embeddings from the existing file
        with open(embedding_file_path, "r") as f:
            embeddings = json.load(f)
    else:
        print(f"Embedding file {filename}.json not found. Generating new embeddings...")
        # If file doesn't exist, generate embeddings
        embeddings = [
            get_embeddings_from_ollama(modelname, chunk) for chunk in chunks
        ]
        # Save the embeddings to file
        save_embeddings(filename, embeddings)
    
    return embeddings

# Simpan embedding ke file
def save_embeddings(filename, embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)


# # Mendapatkan embedding
# def get_embeddings(filename, modelname, chunks):
#     embeddings = [
#         get_embeddings_from_ollama(modelname, chunk) for chunk in chunks
#     ]
#     save_embeddings(filename, embeddings)
#     return embeddings

# # Simpan embedding ke file
# def save_embeddings(filename, embeddings):
#     if not os.path.exists("embeddings"):
#         os.makedirs("embeddings")
#     with open(f"embeddings/{filename}.json", "w") as f:
#         json.dump(embeddings, f)

# Cosine similarity untuk menemukan kemiripan
def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)


# Streamlit App
# Fungsi utama
def main():
    SYSTEM_PROMPT = """You are an assistant that answers questions only in Bahasa Indonesia. 
    Your answers must be based solely on the provided context extracted from the documents. 
    If the answer cannot be determined from the context, respond with "Maaf, saya tidak tahu." 
    Do not include any information outside of the given context, and strictly reply in Bahasa Indonesia.

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
    embeddings = get_embeddings("data_api", "nomic-embed-text", all_paragraphs)

    # Streamlit UI
    st.title("Chatbot dengan RAG (Retrieval-Augmented Generation)")
    st.write("Ajukan pertanyaan berdasarkan data yang ada di folder `data`.")

    # Menyimpan riwayat percakapan
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input pengguna
    user_input = st.text_input("Pertanyaan Anda:", key="input")
    if st.button("Kirim") and user_input.strip():
        # Proses pertanyaan
        # prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=user_input)["embedding"]
        prompt_embedding = get_embeddings_from_ollama("nomic-embed-text", chunk=user_input)
        
        # Persiapkan konteks untuk chat
        most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

        # Persiapkan konteks untuk chat
        context = "\n".join(all_paragraphs[item[1]] for item in most_similar_chunks)

        # Generate response
        response = chat_with_ollama(
            model="llama3",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + context},
                {"role": "user", "content": user_input},
            ]
        )

        # Simpan ke riwayat
        st.session_state.chat_history.append({"user": user_input, "bot": response})

    # Tampilkan riwayat percakapan
    st.subheader("Riwayat Percakapan")
    for chat in st.session_state.chat_history:
        st.markdown(f"**Anda:** {chat['user']}")
        st.markdown(f"**Bot:** {chat['bot']}")

if __name__ == "__main__":
    main()
