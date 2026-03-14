import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv.ipython import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

prompt_template = """
Answer the following question based only on the provided context:
<context>
    {context}
</context>
<question>
    {input}
</question>
"""

llm = ChatOpenAI(model="gpt-4o", temperature=0)


CYBORG_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Rajdhani:wght@400;500;700&display=swap');

:root,
[data-theme="dark"],
[data-theme="light"] {
    --cy-bg-0: #06080d;
    --cy-bg-1: #0b111a;
    --cy-bg-2: #111b2a;
    --cy-panel: rgba(15, 24, 37, 0.82);
    --cy-panel-2: rgba(10, 16, 26, 0.92);
    --cy-border: rgba(78, 217, 255, 0.33);
    --cy-text: #dce7ff;
    --cy-soft: #9ab5d6;
    --cy-accent: #24d9ff;
    --cy-accent-2: #00ffa6;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background:
        radial-gradient(900px 500px at 4% -10%, rgba(36, 217, 255, 0.18), transparent 55%),
        radial-gradient(800px 420px at 100% 0%, rgba(0, 255, 166, 0.10), transparent 60%),
        linear-gradient(160deg, var(--cy-bg-0) 0%, var(--cy-bg-1) 40%, var(--cy-bg-2) 100%) !important;
    color: var(--cy-text) !important;
    font-family: 'Rajdhani', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

h1,
h2,
h3,
h4 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--cy-accent) !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(10, 16, 26, 0.95), rgba(12, 20, 32, 0.92)) !important;
    border-right: 1px solid var(--cy-border) !important;
    box-shadow: inset -1px 0 0 rgba(36, 217, 255, 0.15);
}

[data-testid="stSidebar"] * {
    color: var(--cy-text) !important;
}

[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid var(--cy-border);
    box-shadow: 0 0 0 1px rgba(36, 217, 255, 0.20), 0 8px 28px rgba(0, 0, 0, 0.35);
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stFileUploader"] section,
div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border: 1px solid var(--cy-border) !important;
    background: var(--cy-panel) !important;
    color: var(--cy-text) !important;
    box-shadow: 0 0 0 1px rgba(36, 217, 255, 0.08) !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--cy-accent) !important;
    box-shadow: 0 0 0 1px rgba(36, 217, 255, 0.20), 0 0 16px rgba(36, 217, 255, 0.18) !important;
}

.stButton > button {
    border-radius: 10px !important;
    border: 1px solid rgba(36, 217, 255, 0.65) !important;
    background: linear-gradient(135deg, rgba(13, 36, 56, 0.98), rgba(8, 26, 44, 0.98)) !important;
    color: var(--cy-accent) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
    box-shadow: 0 0 0 1px rgba(36, 217, 255, 0.14), 0 8px 22px rgba(3, 8, 16, 0.45);
}

.stButton > button:hover {
    transform: translateY(-1px);
    border-color: var(--cy-accent-2) !important;
    color: var(--cy-accent-2) !important;
    box-shadow: 0 0 0 1px rgba(0, 255, 166, 0.20), 0 0 20px rgba(0, 255, 166, 0.18);
}

[data-testid="stMarkdownContainer"] p,
label,
[data-testid="stCaptionContainer"],
.stTextInput label,
.stFileUploader label {
    color: var(--cy-soft) !important;
}

[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px;
    background: linear-gradient(180deg, var(--cy-panel), var(--cy-panel-2)) !important;
    border: 1px solid var(--cy-border) !important;
    box-shadow: 0 12px 30px rgba(2, 7, 16, 0.45), inset 0 0 0 1px rgba(36, 217, 255, 0.10);
    padding: 0.7rem 0.85rem;
}

[data-testid="stSpinner"] div {
    border-top-color: var(--cy-accent) !important;
}
</style>
"""


def main():
    st.set_page_config(page_title="RAG", layout="wide")
    st.markdown(CYBORG_CSS, unsafe_allow_html=True)
    st.subheader("Retrieval Augmented generation", divider="blue")

    with st.sidebar:
        st.sidebar.title("Data loader")
        st.image("rag.png")
        pdf_docs = st.file_uploader(label="Load your pdfs", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Loading"):
                content = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        content += page.extract_text()

                splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=512, chunk_overlap=16
                )

                chunks = splitter.split_text(content)
                st.write(chunks)

                embedding_model = OpenAIEmbeddings()
                vector_store = Chroma.from_texts(
                    chunks,
                    embedding_model,
                    collection_name="data_collection",
                )
                retriever = vector_store.as_retriever(
                    kwargs={"k": 5},
                )

                st.session_state.retriever = retriever
    st.subheader("Chatbot")
    user_question = st.text_input("Ask Your Question")
    if user_question:
        context_docs = st.session_state.retriever.invoke(user_question)
        context_list = [d.page_content for d in context_docs]
        context_text = ". ".join(context_list)
        # st.write(context_text)
        prompt = prompt_template.format(context=context_text, input=user_question)

        resp = llm.invoke(prompt)

        st.write(resp.content)


if __name__ == "__main__":
    main()
