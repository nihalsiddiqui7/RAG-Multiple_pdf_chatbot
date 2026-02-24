import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


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





def main():
    load_dotenv()
    st.set_page_config(page_title="Chat With PDFs", page_icon=":books:",layout="centered")

    st.header("Chat With PDFs :books:")
    st.text_input("Ask a question about your PDF")
    
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDF files here", type=["pdf"], accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Processing..."):
                #get pdf text
                raw_text = get_pdf_texts(pdf_docs)
                


                #create chunks
                text_chunks = get_text_chunks(raw_text)
                

                #create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.success("PDFs processed successfully!")




if __name__ == "__main__":
    main()