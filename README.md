# AUTOSAR RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) assistant for AUTOSAR documentation built using LangChain, ChromaDB, Gemini, and Streamlit.

The system enables users to ask questions about AUTOSAR specifications and receive context-aware answers grounded in official AUTOSAR documentation.

## Features

### Document Processing Pipeline

* Loads AUTOSAR specification documents (PDFs)
* Splits documents into semantic chunks
* Generates vector embeddings using Sentence Transformer Embeddings
* Stores embeddings in ChromaDB

### Hybrid Retrieval

* Vector similarity search using ChromaDB
* Keyword search using BM25
* Ensemble retrieval for improved relevance
* Top-k document chunk retrieval

### Conversational AI

* Gemini 2 Flash for response generation
* Context-aware question answering
* Multi-turn conversation support
* Chat history memory

### Knowledge Grounding

* Answers generated from retrieved AUTOSAR documentation
* Reduces hallucinations through Retrieval-Augmented Generation
* Provides responses based on source documents

### Persistent Storage

* ChromaDB vector database persistence
* Reusable document index
* Fast retrieval without re-embedding documents


## Tech Stack

* Python
* LangChain
* Google Gemini 2 Flash
* Sentence Transformer Embeddings
* ChromaDB
* BM25 Retrieval
* Streamlit


## Learning Objectives

This project demonstrates key RAG concepts:

* Document Loading
* Text Chunking
* Embedding Generation
* Vector Databases
* Similarity Search
* BM25 Retrieval
* Hybrid Retrieval
* Ensemble Retrieval
* LangChain Runnables
* Prompt Engineering
* Conversational Memory
* Retrieval-Augmented Generation (RAG)
