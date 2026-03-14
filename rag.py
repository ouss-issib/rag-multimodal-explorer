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
</quesrtion>
"""

llm = ChatOpenAI(model="gpt-4o", temperature=0)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;500;700&display=swap');

:root {
    --bg-start: #0e1b1a;
    --bg-mid: #123a3f;
    --bg-end: #f1b94a;
    --surface: rgba(10, 20, 26, 0.66);
    --surface-strong: rgba(8, 16, 22, 0.86);
    --text-main: #f4f6f8;
    --text-soft: #d7e2e7;
    --accent: #ff7d2a;
    --accent-2: #3dd6b5;
    --border: rgba(255, 255, 255, 0.18);
}

.stApp {
    background:
        radial-gradient(circle at 12% 14%, rgba(61, 214, 181, 0.28), transparent 24%),
        radial-gradient(circle at 84% 18%, rgba(255, 125, 42, 0.24), transparent 26%),
        linear-gradient(135deg, var(--bg-start) 0%, var(--bg-mid) 52%, var(--bg-end) 115%);
    color: var(--text-main);
    font-family: 'Manrope', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(5, 14, 19, 0.94), rgba(8, 34, 39, 0.82));
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text-main) !important;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.02em;
}

[data-testid="stTextInput"] input,
[data-testid="stFileUploader"] section,
.stButton > button {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text-main) !important;
    backdrop-filter: blur(6px);
}

.stButton > button {
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent) 0%, #ffad4e 100%) !important;
    border: none !important;
    color: #111 !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.24);
}

[data-testid="stMarkdownContainer"] p,
label,
.stTextInput label {
    color: var(--text-soft) !important;
}

[data-testid="stImage"] img {
    border-radius: 16px;
    border: 1px solid var(--border);
}

[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface-strong);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 0.6rem 0.8rem;
}
</style>
"""


def main():
    st.set_page_config(page_title="RAG", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.subheader("Retrieval Augmented generation", divider="orange")

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
    st.subheader("Chatbot", divider="rainbow")
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
