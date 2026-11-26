
from sentence_transformers import SentenceTransformer
import numpy as np

from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE


class Embedder:
    """
    Generates embeddings for PDF chunks and user questions
    using Sentence Transformers.
    """

    def __init__(self):
        """Load the embedding model once."""

        print(f"Loading embedding model: {EMBEDDING_MODEL}")

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

        if self.embedding_dimension is None:
            raise RuntimeError(
                "Could not determine embedding model dimension."
            )

        print("Embedding model loaded successfully!")
        print(f"Embedding dimension: {self.embedding_dimension}")

    def embed_chunks(self, chunks):
        """
        Generate embeddings for a collection of PDF chunks.

        Args:
            chunks (list): List of chunk dictionaries.

        Returns:
            numpy.ndarray: Normalized chunk embeddings.
        """

        if not chunks:
            raise ValueError("Cannot generate embeddings for empty chunks.")

        chunk_texts = []

        for chunk in chunks:
            if not isinstance(chunk, dict) or "text" not in chunk:
                raise ValueError(
                    "Each chunk must be a dictionary containing 'text'."
                )

            text = chunk["text"]

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "Chunk text must be a non-empty string."
                )

            chunk_texts.append(text)

        embeddings = self.model.encode(
            chunk_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.shape != (
            len(chunks),
            self.embedding_dimension
        ):
            raise RuntimeError(
                f"Unexpected embedding shape: {embeddings.shape}"
            )

        return embeddings

    def embed_query(self, question):
        """
        Generate an embedding for a user question.

        Args:
            question (str): User's question.

        Returns:
            numpy.ndarray: Normalized question embedding.
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        embedding = self.model.encode(
            question.strip(),
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.shape != (self.embedding_dimension,):
            raise RuntimeError(
                f"Unexpected query embedding shape: {embedding.shape}"
            )

        return embedding
