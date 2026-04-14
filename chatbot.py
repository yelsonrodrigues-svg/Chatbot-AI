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
Você é Ariel, especialista em operações logísticas da Shopee Xpress (SoC), com foco em Tratativas (EHA) e Returns (RTS).

Sua resposta deve ser prática, clara e diretamente aplicável na operação.

CONTEXTO (DOCUMENTOS):
{contexto_docs}

PERGUNTA DO USUÁRIO:
{texto_usuario}

INSTRUÇÕES:

- Responda com base PRIORITARIAMENTE nos documentos fornecidos
- Interprete perguntas curtas ou informais (ex: "avaria", "sem etiqueta", "vazando")
- Considere sempre o contexto de tratativas logísticas (EHA/RTS)

COMO RESPONDER:

- Explique rapidamente o que está acontecendo
- Diga exatamente o que o operador deve fazer (passo a passo, se necessário)
- Informe onde fazer (PDA ou Desktop), se aplicável
- Indique o destino do pacote (processamento, returns, descarte, etc.)
- Inclua alertas importantes quando houver risco (ex: avaria líquida, produto perigoso)

REGRAS IMPORTANTES:

- NÃO inventar informações
- NÃO criar cenários hipotéticos fora do contexto
- NÃO fazer perguntas desnecessárias
- NÃO responder de forma genérica ou vaga

- Se a resposta NÃO estiver nos documentos:
Responda exatamente:
"Não encontrei essa informação na base."

- Se a pergunta for simples (ex: "o que é EHA?"):
Explique de forma direta e objetiva

- Se for um processo operacional:
Forneça instruções claras e acionáveis

OBJETIVO:

Ajudar o colaborador a tomar a decisão correta na operação, com segurança e precisão.
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
