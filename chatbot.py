import streamlit as st
import base64
import os

from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings

# =========================
# 🔐 GROQ CONFIG
# =========================
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

MODEL_NAME = "llama-3.1-8b-instant"

# =========================
# ⚙️ STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Shopee AI",
    page_icon="logo.shopee.png",
    layout="centered"
)

# =========================
# 🖼️ IMAGE BASE64
# =========================
def imagem_base64(caminho):
    if not os.path.exists(caminho):
        return ""
    with open(caminho, "rb") as img:
        return base64.b64encode(img.read()).decode()

# =========================
# 📚 RAG BASE
# =========================
@st.cache_resource(show_spinner=False)
def carregar_base_conhecimento():
    caminho_indice = "base_faiss"
    pasta_docs = "documentos"

    embeddings = CohereEmbeddings(
        cohere_api_key=st.secrets["COHERE_API_KEY"],
        model="embed-multilingual-v3.0"
    )

    if os.path.exists(caminho_indice):
        try:
            db = FAISS.load_local(
                caminho_indice,
                embeddings,
                allow_dangerous_deserialization=True
            )
            _ = db.similarity_search("teste", k=1)
            return db
        except Exception:
            import shutil
            shutil.rmtree(caminho_indice)

    documentos = []

    if not os.path.exists(pasta_docs):
        return None

    for arquivo in os.listdir(pasta_docs):
        if arquivo.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pasta_docs, arquivo))
            documentos.extend(loader.load())

    if not documentos:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs_divididos = splitter.split_documents(documentos)
    db = FAISS.from_documents(docs_divididos, embeddings)
    db.save_local(caminho_indice)

    return db

# =========================
# 🚀 LOAD DB
# =========================
base_conhecimento = carregar_base_conhecimento()

# =========================
# 🎨 BANNER
# =========================
img_base64 = imagem_base64("shopee.work.png")
logo_base64 = imagem_base64("logo.shopee.png")

st.markdown(f"""
<div style="
    position: relative;
    height: 140px;
    border-radius: 18px;
    overflow: hidden;
    margin-bottom: 15px;
    background-image: url('data:image/png;base64,{img_base64}');
    background-size: cover;
    background-position: center;
">
    <div style="display:flex;align-items:center;gap:12px;padding:35px;">
        <img src="data:image/png;base64,{logo_base64}" width="60">
        <h2 style="color:black;">Chatbot EHA & Returns</h2>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# 💬 CHAT MEMORY
# =========================
if "lista_mensagens" not in st.session_state:
    st.session_state.lista_mensagens = []

for msg in st.session_state.lista_mensagens:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

texto_usuario = st.chat_input("Digite sua pergunta sobre EHA ou Returns...")

# =========================
# 🤖 CHAT GROQ + RAG
# =========================
if texto_usuario:

    st.session_state.lista_mensagens.append(
        {"role": "user", "content": texto_usuario}
    )

    with st.chat_message("user"):
        st.markdown(texto_usuario)

    try:
        # 🔎 RAG SEARCH
        if base_conhecimento:
            docs_relacionados = base_conhecimento.similarity_search(texto_usuario, k=4)
            contexto_docs = "\n\n".join([d.page_content[:1000] for d in docs_relacionados])
        else:
            contexto_docs = "Base de conhecimento vazia."

        # 🧠 PROMPT FINAL
        prompt_final = f"""
Você é Ariel, um assistente especialista em processos logísticos da Shopee.

Responda SOMENTE com base nos documentos abaixo.
Se não encontrar a resposta, diga: "Não encontrei essa informação na base."

DOCUMENTOS:
{contexto_docs}

PERGUNTA:
{texto_usuario}
"""

        # 🚀 GROQ REQUEST
        chat_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um assistente especialista em logística e processos da Shopee."
                },
                {
                    "role": "user",
                    "content": prompt_final
                }
            ],
            temperature=0.3,
            max_tokens=800
        )

        texto_resposta = chat_completion.choices[0].message.content

    except Exception as e:
        st.error("❌ Erro ao gerar resposta")
        st.code(str(e))
        texto_resposta = "Erro ao processar sua pergunta."

    with st.chat_message("assistant"):
        st.markdown(texto_resposta)

    st.session_state.lista_mensagens.append(
        {"role": "assistant", "content": texto_resposta}
    )

# =========================
# 🧹 CLEAR CHAT
# =========================
if st.button("🧹 Limpar conversa"):
    st.session_state.lista_mensagens = []
    st.rerun()

