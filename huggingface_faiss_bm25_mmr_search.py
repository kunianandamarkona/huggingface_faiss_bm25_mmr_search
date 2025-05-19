import os
import shutil  # For deleting directories
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage  # For passing user queries
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text
from langchain_pymupdf4llm import PyMuPDF4LLMLoader  # For PDF processing


class VectorDatabase:
    def __init__(self, pdf_path: str, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the VectorDatabase with the given PDF path and embedding model.
        """
        self.pdf_path = pdf_path
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self.index_path = "faiss_index"  # Directory to store FAISS index
        self._cached_docs = None  # Cache for loaded documents

    def _load_pdf(self) -> list:
        """
        Load and split PDF documents into chunks. Use cached documents if available.
        """
        if self._cached_docs is not None:
            return self._cached_docs  # Return cached documents if available

        if not self.pdf_path:
            raise RuntimeError("PDF path is not specified, and documents cannot be loaded.")

        try:
            print(f"Loading PDF from: {self.pdf_path}")
            loader = PyMuPDF4LLMLoader(self.pdf_path, extract_images=False, mode="page")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False
            )
            self._cached_docs = text_splitter.split_documents(docs)  # Cache the documents
            return self._cached_docs
        except Exception as e:
            raise RuntimeError(f"Error processing PDF: {str(e)}")

    def create_or_load_vector_store(self):
        """
        Create a new FAISS vector database or load an existing one.
        """
        if os.path.exists(self.index_path):
            print("Loading existing FAISS index...")
            return FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            if not self.pdf_path:
                raise RuntimeError("PDF path must be specified to create a new FAISS index.")
            print("Creating a new FAISS index...")
            docs = self._load_pdf()
            vector_store = FAISS.from_documents(docs, self.embeddings)
            vector_store.save_local(self.index_path)  # Save the FAISS index
            print(f"FAISS index created and saved at: {self.index_path}")
            return vector_store

    def reset_faiss_index(self):
        """
        Delete the FAISS index directory to reset the vector database.
        """
        if os.path.exists(self.index_path):
            print(f"Deleting FAISS index at: {self.index_path}")
            shutil.rmtree(self.index_path)
        else:
            print("FAISS index does not exist. No action taken.")

    def get_documents(self):
        """
        Return cached documents. If documents are not cached, attempt to load them
        (only applicable if the PDF file path is specified).
        """
        if self._cached_docs is not None:
            return self._cached_docs
        elif self.pdf_path:
            return self._load_pdf()
        else:
            raise RuntimeError("No cached documents available, and PDF path is not specified.")


class HybridSearch:
    def __init__(self, pdf_path: str, embedding_model: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the HybridSearch system by loading the vector database,
        setting up the hybrid retriever, and configuring the LLM.
        """
        # Initialize the vector database with the PDF path
        self.vector_db = VectorDatabase(pdf_path=pdf_path, embedding_model=embedding_model)

        # Reset FAISS index (optional: only if you want to clear it)
        self.vector_db.reset_faiss_index()

        # Create or load FAISS index
        self.vector_store = self.vector_db.create_or_load_vector_store()

        # Set up the language model (CXAI Playground LLM)
        api_token = os.getenv("CXAI_PLAYGROUND_ACCESS_TOKEN")
        if not api_token:
            raise ValueError("API token is missing. Please add it to the .env file.")
        self.llm = ChatOpenAI(
            model="o4-mini",
            base_url="https://cxai-playground.cisco.com",
            openai_api_key=api_token,
            temperature=1  # Set temperature to 1 to avoid model-specific errors
        )

        # Initialize the hybrid retriever
        self.hybrid_retriever = self._create_multi_retriever()

    def _create_multi_retriever(self):
        """
        Create a multi-strategy retriever combining:
        1. Similarity Search
        2. Maximum Marginal Relevance (MMR)
        3. BM25 Retriever
        """
        # Similarity Search Retriever
        similarity_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Maximum Marginal Relevance (MMR) Retriever
        mmr_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
        )

        # BM25 Retriever
        docs = self.vector_db.get_documents()  # Use cached documents or load them if needed
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 5

        # Combine all retrievers into an EnsembleRetriever
        return EnsembleRetriever(
            retrievers=[similarity_retriever, mmr_retriever, bm25_retriever],
            weights=[0.4, 0.4, 0.2]
        )

    def query(self, question: str) -> dict:
        """
        Query the system with a user question and use the LLM to generate an answer.
        """
        try:
            if not isinstance(question, str):
                raise ValueError("The question must be a string.")
            
            # Retrieve relevant documents using the hybrid retriever
            retrieved_docs = self.hybrid_retriever.invoke(question)
            sources = [doc.page_content for doc in retrieved_docs]

            # Combine the retrieved sources into a context for the LLM
            context = "\n\n".join(sources[:3])  # Use the top 3 sources as context
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

            # Use the LLM to generate an answer
            response = self.llm([HumanMessage(content=prompt)])
            answer = response.content

            # Only return the answer (exclude sources)
            return {
                "answer": answer
            }
        except Exception as e:
            raise RuntimeError(f"Error during query execution: {str(e)}")


if __name__ == "__main__":
    # Define the PDF file path here
    pdf_path = "/Users/snanda2/Desktop/Cisco_rag/Cisco_API/ASR9K_QOS.pdf"

    # Check if the PDF file exists
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' does not exist. Please provide a valid path.")
    else:
        try:
            # Initialize the hybrid search system with the PDF path
            hybrid_search = HybridSearch(pdf_path=pdf_path)

            print("\nWelcome to the Hybrid Multi-Search System!")
            print("Type 'exit' to quit the program.\n")

            while True:
                question = input("Enter your question: ").strip()
                if question.lower() == "exit":
                    print("\nGoodbye!")
                    break
                try:
                    output = hybrid_search.query(question)
                    print("\nAnswer:\n", output["answer"])
                except Exception as e:
                    print(f"An error occurred: {e}")
        except Exception as e:
            print(f"An error occurred during initialization: {e}")
