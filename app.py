import streamlit as st
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from tempfile import NamedTemporaryFile

# Streamlit page title
st.title("RAG-Driven QA System Based on Streamlit and LangChain")

# Input OpenAI API key
api_key = st.text_input("Please enter your OpenAI API key:", type="password")

if api_key:
    # Set OpenAI API key as an environment variable
    os.environ["OPENAI_API_KEY"] = api_key

    # User uploads document
    uploaded_file = st.file_uploader("Upload your document for QA", type=["txt"])

    if uploaded_file is not None:
        # Save the uploaded file as a temporary file
        with NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        # Load the document using the temporary file path
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # Use OpenAI Embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        # Create FAISS vector store
        vector_store = FAISS.from_documents(docs, embeddings)

        # Set the retriever
        retriever = vector_store.as_retriever()

        # Set OpenAI LLM (using ChatOpenAI)
        llm = ChatOpenAI(model_name="gpt-4o-mini")

        # Load the QA chain
        qa_chain = load_qa_chain(llm, chain_type="stuff")

        # User inputs query
        user_query = st.text_input("Please enter your question:")

        if user_query:
            # Retrieve relevant documents
            relevant_docs = retriever.get_relevant_documents(user_query)

            # Use the generative model to answer the question
            answer = qa_chain.run(input_documents=relevant_docs, question=user_query)

            # Display the answer
            st.write("Answer:", answer)
