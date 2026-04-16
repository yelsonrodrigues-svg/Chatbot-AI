import streamlit as st
import base64
import os
import shutil
import re

from groq import Groq

from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
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
# 🧠 CLASSIFICAÇÃO DE MENSAGEM
# =========================
def normalizar_texto(texto: str) -> str:
    texto = texto.strip().lower()
    texto = re.sub(r"\s+", " ", texto)
    return texto

def classificar_mensagem(texto: str) -> str:
    texto_norm = normalizar_texto(texto)

    saudacoes_exatas = {
        "oi",
        "ola",
        "olá",
        "opa",
        "e ai",
        "e aí",
        "bom dia",
        "boa tarde",
        "boa noite",
        "tudo bem",
        "blz",
        "beleza",
        "hello",
        "hi"
    }

    if texto_norm in saudacoes_exatas:
        return "SAUDACAO"

    palavras_operacionais = [
        "pacote", "pacotes", "avaria", "avariado", "motivo", "fluxo", "returns",
        "return", "tratativa", "tratativas", "pda", "desktop", "br", "soc",
        "eha", "etiqueta", "sem etiqueta", "sem identificação", "identificação",
        "recebimento", "reversa", "3pl", "lm_hub", "fm_hub", "ticket", "nf",
        "realocação", "duplicidade", "receita federal", "produto proibido",
        "embalagem", "erro operacional", "offline resolve", "check-in", "check-out"
    ]

    gatilhos_pergunta = [
        "como", "o que", "qual", "quais", "quando", "onde", "por que",
        "porque", "pra onde", "para onde", "devo", "faço", "fazer"
    ]

    if "?" in texto_norm:
        return "PERGUNTA_OPERACIONAL"

    if any(p in texto_norm for p in palavras_operacionais):
        return "PERGUNTA_OPERACIONAL"

    if any(g in texto_norm for g in gatilhos_pergunta) and len(texto_norm.split()) >= 3:
        return "PERGUNTA_OPERACIONAL"

    if len(texto_norm.split()) >= 5:
        return "PERGUNTA_OPERACIONAL"

    return "CONVERSA"

# =========================
# 📚 FUNÇÃO AUXILIAR LOADERS
# =========================
def carregar_documento(caminho_arquivo, nome_arquivo):
    nome_lower = nome_arquivo.lower()

    # PDF
    if nome_lower.endswith(".pdf"):
        try:
            loader = PyPDFLoader(caminho_arquivo)
            return loader.load()
        except Exception as e:
            st.warning(f"Erro ao carregar PDF '{nome_arquivo}': {e}")
            return []

    # CSV
    elif nome_lower.endswith(".csv"):
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                loader = CSVLoader(
                    file_path=caminho_arquivo,
                    encoding=encoding
                )
                return loader.load()
            except Exception:
                continue

        st.warning(f"Erro ao carregar CSV '{nome_arquivo}'. Verifique a codificação do arquivo.")
        return []

    # TXT
    elif nome_lower.endswith(".txt"):
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                loader = TextLoader(
                    caminho_arquivo,
                    encoding=encoding
                )
                return loader.load()
            except Exception:
                continue

        st.warning(f"Erro ao carregar TXT '{nome_arquivo}'. Verifique a codificação do arquivo.")
        return []

    return []

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

    # Tenta carregar índice existente
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
            if os.path.exists(caminho_indice):
                shutil.rmtree(caminho_indice, ignore_errors=True)

    documentos = []

    if not os.path.exists(pasta_docs):
        st.warning("A pasta 'documentos' não foi encontrada.")
        return None

    arquivos = os.listdir(pasta_docs)

    for arquivo in arquivos:
        caminho_arquivo = os.path.join(pasta_docs, arquivo)

        if os.path.isfile(caminho_arquivo):
            docs = carregar_documento(caminho_arquivo, arquivo)
            if docs:
                documentos.extend(docs)

    if not documentos:
        st.warning("Nenhum documento válido encontrado na pasta 'documentos'.")
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
# 🔎 RECUPERA CONTEXTO
# =========================
def buscar_contexto(base_conhecimento, pergunta, k=4):
    if not base_conhecimento:
        return [], ""

    try:
        docs_relacionados = base_conhecimento.similarity_search(pergunta, k=k)
        contexto_docs = "\n\n".join(
            [doc.page_content[:1200] for doc in docs_relacionados if getattr(doc, "page_content", "").strip()]
        )
        return docs_relacionados, contexto_docs
    except Exception:
        return [], ""

# =========================
# 🧠 PROMPT HÍBRIDO
# =========================
def montar_prompt(contexto_docs, texto_usuario, tipo_mensagem):
    return f"""
Você é Ariel, um assistente virtual da Shopee especializado em EHA e Returns.

Seu objetivo é responder de forma natural, útil e confiável.

PRIORIDADES DE COMPORTAMENTO:
1. Entenda a intenção do usuário antes de responder.
2. Se for uma saudação simples, responda de forma natural e breve.
3. Se for uma pergunta operacional sobre processos, regras, tratativas, motivos, avarias, PDA, desktop, BR, Returns, EHA ou operação logística, use os documentos recuperados como fonte principal.
4. Quando a resposta operacional estiver claramente apoiada nos documentos, responda de forma objetiva e fácil de entender.
5. Quando não houver base suficiente para responder uma pergunta operacional, diga exatamente:
"Não encontrei essa informação na base."

DIRETRIZES IMPORTANTES:
- Responda sempre em português do Brasil.
- Não invente processos, regras, menus ou decisões operacionais.
- Você pode reescrever o conteúdo de forma mais clara e natural.
- Não responda com apresentação do assistente quando o usuário fizer uma pergunta operacional.
- Evite começar toda resposta com "Olá" ou se apresentar, exceto quando for realmente uma saudação.
- Evite listas numeradas ou opções 1, 2, 3, a menos que isso ajude muito e esteja claramente apoiado no contexto.
- Se a mensagem for uma dúvida, problema, pergunta de processo ou frase com contexto operacional, trate como pergunta operacional.

TIPO DA MENSAGEM:
{tipo_mensagem}

DOCUMENTOS RECUPERADOS:
{contexto_docs if contexto_docs.strip() else "Nenhum documento recuperado."}

MENSAGEM DO USUÁRIO:
{texto_usuario}
"""

# =========================
# 🚀 LOAD DB
# =========================
base_conhecimento = carregar_base_conhecimento()

# =========================
# 🎨 BANNER
# =========================
img_base64 = imagem_base64("shopee.work.png")
logo_base64 = imagem_base64("logo.shopee.png")

st.markdown(
    f"""
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
    """,
    unsafe_allow_html=True
)

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
# 🤖 CHAT GROQ + RAG HÍBRIDO
# =========================
if texto_usuario:
    st.session_state.lista_mensagens.append(
        {"role": "user", "content": texto_usuario}
    )

    with st.chat_message("user"):
        st.markdown(texto_usuario)

    try:
        tipo_mensagem = classificar_mensagem(texto_usuario)

        if tipo_mensagem == "SAUDACAO":
            contexto_docs = ""
        elif tipo_mensagem == "CONVERSA":
            contexto_docs = ""
        else:
            _, contexto_docs = buscar_contexto(base_conhecimento, texto_usuario, k=4)

        prompt_final = montar_prompt(
            contexto_docs=contexto_docs,
            texto_usuario=texto_usuario,
            tipo_mensagem=tipo_mensagem
        )

        chat_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é Ariel, um assistente confiável, natural e objetivo. "
                        "Você entende a intenção do usuário antes de responder. "
                        "Você responde saudações de forma breve. "
                        "Você responde perguntas operacionais com base no contexto fornecido. "
                        "Você nunca se apresenta desnecessariamente em perguntas operacionais. "
                        "Você nunca inventa processos, regras, fluxos ou menus."
                    )
                },
                {
                    "role": "user",
                    "content": prompt_final
                }
            ],
            temperature=0.2,
            max_tokens=800
        )

        texto_resposta = chat_completion.choices[0].message.content.strip()

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
