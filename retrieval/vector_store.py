
import faiss
import numpy as np


class VectorStore:
    """
    Manages FAISS vector indexing and similarity search.

    Uses IndexFlatIP with normalized embeddings
    to perform cosine similarity search.
    """

    def __init__(self, embedding_dimension):
        """
        Initialize the FAISS index.

        Args:
            embedding_dimension (int): Dimension of each embedding.
        """

        if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
            raise ValueError(
                "Embedding dimension must be a positive integer."
            )

        self.embedding_dimension = embedding_dimension

        # Create an exact inner-product FAISS index.
        self.index = faiss.IndexFlatIP(embedding_dimension)

        # Stores the original chunk records in vector insertion order.
        self.chunks = []

        print("FAISS vector index initialized successfully!")
        print(f"Embedding dimension: {embedding_dimension}")

    def add_embeddings(self, embeddings, chunks):
        """
        Add document embeddings and their corresponding metadata.

        Args:
            embeddings (numpy.ndarray): Document embeddings.
            chunks (list): Original chunk dictionaries.
        """

        if len(chunks) == 0:
            raise ValueError("Cannot index an empty chunk collection.")

        if self.index.ntotal != 0 or self.chunks:
            raise RuntimeError(
                "This vector store already contains data. "
                "Create a new VectorStore for a new indexing operation."
            )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2-dimensional array."
            )

        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dimension}, "
                f"received {embeddings.shape[1]}."
            )

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                "Number of embeddings must match number of chunks."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain invalid numerical values.")

        # Check that embeddings are normalized.
        norms = np.linalg.norm(embeddings, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError(
                "Embeddings must be normalized before indexing."
            )

        # Validate metadata before modifying the index.
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise TypeError("Each chunk must be a dictionary.")

            required_fields = ("chunk_id", "page_number", "text")

            if not all(field in chunk for field in required_fields):
                raise ValueError(
                    "Each chunk must contain chunk_id, page_number, and text."
                )

        # Insert embeddings into FAISS.
        self.index.add(embeddings)

        # Preserve the matching chunk records.
        self.chunks = list(chunks)

        print("\nEmbeddings indexed successfully!")
        print(f"Total vectors indexed: {self.index.ntotal}")

    def search(self, query_embedding, top_k=5):
        """
        Retrieve the most relevant chunks for a query vector.

        Args:
            query_embedding (numpy.ndarray): Normalized query embedding.
            top_k (int): Maximum number of results.

        Returns:
            list: Retrieved chunk records with similarity scores.
        """

        if self.index.ntotal == 0:
            raise RuntimeError("Cannot search an empty FAISS index.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_embedding.shape == (self.embedding_dimension,):
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape != (1, self.embedding_dimension):
            raise ValueError(
                f"Expected query shape (1, {self.embedding_dimension}), "
                f"received {query_embedding.shape}."
            )

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding contains invalid values.")

        query_norm = np.linalg.norm(query_embedding)

        if not np.isclose(query_norm, 1.0, atol=1e-4):
            raise ValueError(
                "Query embedding must be normalized."
            )

        # Do not request more results than available vectors.
        actual_k = min(top_k, self.index.ntotal)

        # Search the FAISS index.
        scores, indices = self.index.search(
            query_embedding,
            actual_k
        )

        results = []

        for score, vector_id in zip(scores[0], indices[0]):

            # FAISS uses -1 for missing results.
            if vector_id == -1:
                continue

            # Retrieve the original chunk using its vector position.
            chunk = self.chunks[int(vector_id)]

            result = chunk.copy()

            # Attach similarity information.
            result["similarity_score"] = float(score)
            result["vector_id"] = int(vector_id)

            results.append(result)

        return results

    def get_index_size(self):
        """Return the total number of indexed vectors."""

        return self.index.ntotal
