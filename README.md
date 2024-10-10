# RAG-Driven-QA-System-Based-on-Streamlit-and-LangChain

This project implements a **Retrieval-Augmented Generation (RAG)**-based Question Answering (QA) system using **Streamlit** for the front-end interface and **LangChain** for backend processing. The system allows users to upload documents and ask questions, with answers generated based on relevant sections of the uploaded content.

## Features

- **Document Upload**: Users can upload text files for analysis.
- **Document Chunking**: The system processes documents by splitting them into chunks for better handling and retrieval.
- **Efficient Retrieval**: FAISS is used for fast retrieval of relevant document chunks.
- **OpenAI GPT-4o-mini**: The generative model (via OpenAI’s GPT-4o-mini) provides context-aware answers based on the document content.
- **User-Friendly Interface**: Built with Streamlit, the app offers a simple and interactive interface for seamless question-answering.

## Requirements

Before running the application, ensure you have the following dependencies installed:

- Python 3.8+
- Streamlit
- LangChain
- FAISS
- OpenAI API Key

You can install the required packages using:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```