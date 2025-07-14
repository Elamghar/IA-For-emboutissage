import os
import streamlit as st
import google.generativeai as genai

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

# ---------- Configuration ----------
GEMINI_API_KEY = "TA_CLE_API_GEMINI"           # ← remplace par ta clé
DOSSIER_PDFS_PAR_DEFAUT = "pdfs"               # dossier où tu mets tes PDF de base
NB_CHUNKS_REPONSE     = 4                      # nombre de passages renvoyés à Gemini
CHUNK_SIZE            = 500                    # longueur d’un chunk
CHUNK_OVERLAP         = 50

# ---------- Init Gemini ----------
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-pro")

# ---------- Embeddings & Splitter ----------
embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

# ---------- Helpers ----------
def split_pdf(path: str):
    """Charge un PDF → liste de Document (chunks)."""
    docs = PyPDFLoader(path).load()
    return splitter.split_documents(docs)

def charger_pdfs_dossier(dossier: str):
    """Parcourt le dossier et renvoie la liste de tous les chunks."""
    if not os.path.exists(dossier):
        return []
    chunks = []
    for f in os.listdir(dossier):
        if f.lower().endswith(".pdf"):
            chunks.extend(split_pdf(os.path.join(dossier, f)))
            st.sidebar.info(f"📄 PDF chargé : {f}")
    return chunks

def rag_answer(question: str):
    """Recherche contextuelle puis envoie à Gemini."""
    if "vectordb" not in st.session_state or st.session_state.vectordb is None:
        return "❌ Aucun document chargé."

    # 1) Recherche des passages les plus pertinents
    hits = st.session_state.vectordb.similarity_search(question, k=NB_CHUNKS_REPONSE)
    context = "\n\n".join([doc.page_content for doc in hits])

    # 2) Prompt
    prompt = f"""
Tu es un assistant. Voici du contexte extrait de divers documents :

{context}

Question : {question}
Réponds de façon claire et concise en t’appuyant uniquement sur ces documents.
"""
    try:
        rep = gemini.generate_content(prompt)
        return rep.text.strip()
    except Exception as e:
        return f"⚠️ Erreur Gemini : {e}"

# ---------- App Streamlit ----------
st.set_page_config(page_title="Chatbot PDF + Gemini", layout="wide")
st.title("🤖 Chatbot PDF (base par défaut + ajouts utilisateur)")
st.markdown(
    "- Les PDF du dossier **`pdfs/`** sont chargés au démarrage\n"
    "- Tu peux **ajouter d’autres PDF** à tout moment ; ils s’ajouteront à la base de connaissances\n"
)

# --- Initialisation de l’index partagé ---
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
    st.session_state.fichiers = []   # noms des pdf chargés

    # Charger les PDF “base”
    base_chunks = charger_pdfs_dossier(DOSSIER_PDFS_PAR_DEFAUT)
    if base_chunks:
        st.session_state.vectordb = FAISS.from_documents(base_chunks, embedder)
        st.session_state.fichiers.extend(
            [f for f in os.listdir(DOSSIER_PDFS_PAR_DEFAUT) if f.lower().endswith(".pdf")]
        )

# --- Upload utilisateur ---
up = st.file_uploader("📥 Ajouter un PDF", type="pdf")
if up:
    chunks = split_pdf(up.name)
    if st.session_state.vectordb is None:
        st.session_state.vectordb = FAISS.from_documents(chunks, embedder)
    else:
        st.session_state.vectordb.add_documents(chunks)
    st.success(f"✅ {up.name} ajouté.")
    st.session_state.fichiers.append(up.name)

# --- Affichage des PDF actuellement indexés ---
with st.expander("📂 PDF actuellement chargés"):
    for f in st.session_state.fichiers:
        st.write("•", f)

# --- Zone de question ---
question = st.text_input("💬 Pose ta question :")
if st.button("Envoyer") and question:
    reponse = rag_answer(question)
    st.text_area("✏️ Réponse", value=reponse, height=250)

