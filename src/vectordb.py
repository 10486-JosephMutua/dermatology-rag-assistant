import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """

        
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks=splitter.split_text(text)


        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        

        print(f"Processing {len(documents)} documents...")
        ids=[]
        embeddings=self.embedding_model.encode(documents)
        for i, doc in enumerate(documents):
            ids.append(f"doc_{i}")
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids)
        

        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
      
        
        query_embedding=self.embedding_model.encode([query])
        latest_emdedding=query_embedding[-1]
        
       
        try:
           
            results = self.collection.query(
                query_embeddings=latest_emdedding,
                n_results=n_results,
                include=["documents", "metadatas", "distances", "embeddings"]
            )
          
        except Exception as e:
            print(f"Error during collection query: {e}")
            results=None
        if results is None:
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }

       
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])
        ids = results.get("ids", [])  

        documents = documents[0] if documents and isinstance(documents, list) and len(documents) > 0 else []
        metadatas = metadatas[0] if metadatas and isinstance(metadatas, list) and len(metadatas) > 0 else []
        distances = distances[0] if distances and isinstance(distances, list) and len(distances) > 0 else []
        ids = ids[0] if ids and isinstance(ids, list) and len(ids) > 0 else []

        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
            "ids": ids,
        }


        