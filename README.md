# huggingface_faiss_bm25_mmr_search
This script implements a powerful hybrid search system that combines HuggingFace embeddings with multiple retrieval mechanisms, including FAISS-based similarity search, BM25, and Maximum Marginal Relevance (MMR), to enable efficient and accurate semantic search over PDF documents.
Hybrid Search System with HuggingFace, FAISS, and LangChain

Hybrid Search System with HuggingFace, FAISS, and LangChain

This project implements a hybrid search system for querying PDF documents using a combination of HuggingFace embeddings, FAISS, BM25, and MMR retrieval mechanisms. The system allows you to extract relevant information from large PDF documents and generate accurate, context-aware answers using a LangChain-integrated LLM.



Features

Hybrid Retrieval:

Combines three search mechanisms:
FAISS-based Similarity Search
BM25 (Keyword Search)
Maximum Marginal Relevance (MMR)
HuggingFace Embeddings:

Leverages HuggingFace Sentence-Transformers for creating semantic vector representations of document content.
PDF Processing:

Extracts and splits text from PDF documents into manageable chunks for efficient querying.
Vector Database:

Builds and manages a vector database using FAISS for fast similarity search.
Context-Aware Querying:

Retrieves relevant document sections based on user queries and generates detailed responses using a LangChain-integrated LLM.
Interactive System:

Enables users to query PDF documents interactively via the terminal.


Technologies Used

LangChain: For orchestration, retrievers, and LLM integration.
HuggingFace: For generating dense vector embeddings.
FAISS: For fast and scalable vector similarity search.
BM25: For keyword-based retrieval.
PyMuPDF: For extracting and processing text from PDF files.
LangChain LLM: For generating context-aware answers.


Installation

Prerequisites

Python 3.9 or higher.
Pip (Python package manager).

Install Dependencies

Clone the repository or download the script.
Create and activate a virtual environment (optional but recommended):
bash
Copy Code
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
Install the required dependencies:
bash
Copy Code
pip install -r requirements.txt


Usage

Run the Script

Ensure the target PDF file is available. Update the pdf_path variable in the script with the path to your PDF file.

Run the script:

bash
Copy Code
python huggingface_faiss_bm25_mmr_search.py
You'll see the following prompt:

plaintext
Copy Code
Welcome to the Hybrid Multi-Search System!
Type 'exit' to quit the program.

Enter your question:
Enter your questions interactively, and the system will generate a context-aware response based on the content of the PDF document.



Example

Input:

plaintext
Copy Code
Enter your question: What is the recommended DSCP value for voice traffic?

Output:

plaintext
Copy Code
Answer:
The recommended DSCP value for voice traffic is EF (Expedited Forwarding), which corresponds to DSCP value 46. It ensures low latency and high priority for voice packets.


Configuration

Environment Variables

The script requires an API token for LangChain's LLM integration. Add your API token to a .env file in the project directory:


plaintext
Copy Code
CXAI_PLAYGROUND_ACCESS_TOKEN=your_api_token_here

Change Embedding Model

You can change the HuggingFace embedding model by modifying the embedding_model parameter in the script:


python
Copy Code
embedding_model = "sentence-transformers/all-mpnet-base-v2"  # Default

Recommended models:


sentence-transformers/all-mpnet-base-v2 (default)
sentence-transformers/all-MiniLM-L6-v2 (faster, less accurate)
sentence-transformers/multi-qa-mpnet-base-dot-v1 (optimized for Q&A).


File Structure

The project contains the following files and directories:


Copy Code
.
├── huggingface_faiss_bm25_mmr_search.py  # Main script
├── requirements.txt                     # Required dependencies
├── README.md                            # Project documentation
└── .env                                 # Environment variables (optional)


Requirements

Below are the required Python packages listed in requirements.txt:


plaintext
Copy Code
langchain==0.3.25
langchain-community==0.3.24
langchain-huggingface==0.3.16
langchain-pymupdf4llm==0.4.0
langchain-text-splitters==0.3.8
faiss-cpu==1.11.0
sentence-transformers==4.1.0
huggingface-hub==0.30.2
PyMuPDF==1.25.5
pymupdf4llm==0.0.22
rank-bm25==0.2.2
openai==1.77.0


Known Limitations

Large PDFs:
Processing very large PDFs may result in high memory usage.
Default Temperature:
The LLM is configured with a default temperature of 1, which may produce diverse or creative responses.
Token Limit:
Responses are limited by the token limit of the LLM and retriever.


Future Enhancements

Add support for additional retrievers (e.g., TF-IDF).
Implement caching for frequently queried results.
Extend support for multi-file PDF search.


License

This project is licensed under the [MIT License](LICENSE).



