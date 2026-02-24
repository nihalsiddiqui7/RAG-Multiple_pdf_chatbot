import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
import os

def get_pdf_texts(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore



def get_qa_chain(vectorstore):
    prompt_template = """
    You are a helpful assistant answering questions from uploaded PDFs.
    Use ONLY the provided context.If you don't know the answer, say you don't know. Do not make up an answer.
    YOU MUST ALWAYS use the provided context to answer the question. If the question cannot be answered using the provided context, say you don't know.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Allow overriding the model via env; default to a broadly supported Inference model
    model_id = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not api_token:
        st.warning("Set HUGGINGFACEHUB_API_TOKEN in your .env to call Hugging Face Inference.")

    model = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        temperature=0.2,
        max_new_tokens=512,
        huggingfacehub_api_token=api_token,
    )

    chat = ChatHuggingFace(llm=model, verbose=True)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return qa_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat With PDFs", page_icon="📚")

    st.header("Chat With PDFs 📚")

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    question = st.text_input("Ask a question about your PDF")

    if question and st.session_state.qa_chain:
        result = st.session_state.qa_chain.invoke(question)
        st.write("### Answer:")
        st.write(result["result"])

    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_texts(pdf_docs)
                chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(chunks)
                st.session_state.qa_chain = get_qa_chain(vectorstore)

                st.success("Ready to chat!")




if __name__ == "__main__":
    main()