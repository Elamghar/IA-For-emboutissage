import os
import streamlit as st
import requests
from datetime import datetime
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
 
# ---------- Configuration ----------
TOGETHER_API_KEY = "9ce2b5fb09531e1188727691fa9d5c36f6107184251d63c3bac05d08a888b91_4"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DOSSIER_PDFS_PAR_DEFAUT = "pdfs"
NB_CHUNKS_REPONSE = 4
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
 
# ---------- Embeddings & Splitter ----------
embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
 
# ---------- Session State Init ----------
def init_session():
    for key in ['conversation', 'vectordb', 'fichiers']:
        if key not in st.session_state:
            st.session_state[key] = [] if key in ['conversation', 'fichiers'] else None
 
# ---------- PDF Handling ----------
def split_pdf(path: str):
    try:
        docs = PyPDFLoader(path).load()
        return splitter.split_documents(docs)
    except Exception as e:
        st.error(f"Erreur lors du traitement du PDF {path}: {e}")
        return []
 
def charger_pdfs_dossier(dossier: str):
    if not os.path.exists(dossier):
        os.makedirs(dossier)
        return []
    chunks = []
    for f in os.listdir(dossier):
        if f.lower().endswith(".pdf"):
            chunks.extend(split_pdf(os.path.join(dossier, f)))
            st.sidebar.info(f"📄 PDF chargé : {f}")
    return chunks
 
# ---------- Fonction d'appel à l'API Together ----------
def call_mistral(prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Erreur Mistral: {e}"
 
# ---------- RAG Answer ----------
def rag_answer(question: str):
    if st.session_state.vectordb is None:
        return "⚠️ Aucun document chargé."
    hits = st.session_state.vectordb.similarity_search(question, k=NB_CHUNKS_REPONSE)
    context = "\n\n".join([doc.page_content for doc in hits])
    prompt = f"""
Tu es un assistant expert. Voici des extraits de documents :
 
{context}
 
Question : {question}
Réponds de façon claire, concise,court et en t'appuyant uniquement sur le contenu fourni. Si l'information ne s'y trouve pas, dis-le clairement.
"""
    return call_mistral(prompt)
 
# ---------- Streamlit App ----------
def main():
    init_session()
    st.set_page_config(page_title="Mistral PDF Chatbot", layout="wide")
    st.title("🤖 Chatbot PDF avec Mistral")
 
    with st.sidebar:
        st.header("📁 Chargement de PDFs")
        if st.session_state.vectordb is None:
            base_chunks = charger_pdfs_dossier(DOSSIER_PDFS_PAR_DEFAUT)
            if base_chunks:
                st.session_state.vectordb = FAISS.from_documents(base_chunks, embedder)
                st.session_state.fichiers.extend([f for f in os.listdir(DOSSIER_PDFS_PAR_DEFAUT) if f.endswith(".pdf")])
                st.success(f"✅ {len(base_chunks)} extraits chargés")
 
        fichier = st.file_uploader("📄 Ajouter un PDF", type="pdf")
        if fichier:
            path = f"temp_{fichier.name}"
            with open(path, "wb") as f:
                f.write(fichier.read())
            chunks = split_pdf(path)
            if st.session_state.vectordb is None:
                st.session_state.vectordb = FAISS.from_documents(chunks, embedder)
            else:
                st.session_state.vectordb.add_documents(chunks)
            st.session_state.fichiers.append(fichier.name)
            os.remove(path)
            st.success(f"✅ {fichier.name} ajouté")
 
        if st.button("🗑️ Réinitialiser historique"):
            st.session_state.conversation = []
            st.rerun()
 
    st.subheader("📄 PDFs chargés")
    for f in st.session_state.fichiers:
        st.write("•", f)
 
    st.subheader("💬 Discussion avec Moqaddem")
    for role, msg, time in st.session_state.conversation:
        st.markdown(f"**{'👤 Vous' if role == 'user' else '🤖 Mistral'}** _{time}_")
        st.markdown(msg)
        st.markdown("---")
 
    st.subheader("✍️ Posez votre question")
    question = st.text_area("Votre message:", height=100)
    if st.button("📤 Envoyer") and question.strip():
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.conversation.append(("user", question, timestamp))
        with st.spinner("🤔 Mistral réfléchit..."):
            reponse = rag_answer(question)
            st.session_state.conversation.append(("ai", reponse, timestamp))
        st.rerun()
 
    st.subheader("📥 Exporter la conversation")
    if st.session_state.conversation:
        data = json.dumps(st.session_state.conversation, ensure_ascii=False, indent=2)
        st.download_button("📥 Télécharger", data=data, file_name="conversation.json", mime="application/json")
 
if __name__ == "__main__":
    main()
 
