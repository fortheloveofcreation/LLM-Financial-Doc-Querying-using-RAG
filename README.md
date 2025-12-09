## LLM-based Chatbot for Financial Document Query

This project implements a Retrieval-Augmented Generation (RAG) chatbot that enables users to query and analyze financial documents efficiently.

### Features
- **RAG Pipeline:** Built using [LangChain](https://www.langchain.com/) to process and retrieve information from financial documents.  
- **Modeling:** Utilizes **Llama 3.1** as the primary language model and **HuggingFace’s all-MiniLM-L6-v2** for text embeddings.  
- **Document Handling:** Supports loading and preprocessing of PDFs and text files using **PyPDFLoader**, followed by segmentation, embedding creation, and vector database indexing for optimized information retrieval.

### Technologies Used
- Python  
- LangChain  
- Llama 3.1  
- Chroma  
- HuggingFace Transformers  
