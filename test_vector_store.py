
from ingestion.pdf_loader import extract_pdf_text
from ingestion.chunker import chunk_pages

from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore

from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def main():

    print("\nTesting FR-05: FAISS Vector Indexing\n")

    # STEP 1: Extract PDF text
    pages = extract_pdf_text(PDF_PATH)

    if not pages:
        print("PDF extraction failed!")
        return

    # STEP 2: Generate chunks
    chunks = chunk_pages(
        pages,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    if not chunks:
        print("No chunks generated!")
        return

    print(f"Total chunks generated: {len(chunks)}")

    # STEP 3: Generate embeddings
    embedder = Embedder()

    embeddings = embedder.embed_chunks(chunks)

    print("\nEmbedding generation completed.")
    print(f"Embedding shape: {embeddings.shape}")

    # STEP 4: Initialize FAISS
    vector_store = VectorStore(
        embedding_dimension=embedder.embedding_dimension
    )

    # STEP 5: Add embeddings to FAISS
    vector_store.add_embeddings(
        embeddings=embeddings,
        chunks=chunks
    )

    print("\nFAISS Index Statistics")
    print("-" * 50)
    print(f"Total vectors indexed: {vector_store.get_index_size()}")
    print(f"Embedding dimensions: {embedder.embedding_dimension}")

    # STEP 6: Create a sample question
    question = "What is the main subject of this document?"

    print("\nUser question:")
    print(question)

    # STEP 7: Generate question embedding
    query_embedding = embedder.embed_query(question)

    # STEP 8: Search FAISS
    retrieved_chunks = vector_store.search(
        query_embedding=query_embedding,
        top_k=5
    )

    # STEP 9: Display retrieved results
    print("\nTop 5 Retrieved Chunks")
    print("=" * 60)

    for rank, result in enumerate(retrieved_chunks, start=1):

        print(f"\nRANK: {rank}")
        print(f"VECTOR ID: {result['vector_id']}")
        print(f"CHUNK ID: {result['chunk_id']}")
        print(f"SOURCE PAGE: {result['page_number']}")

        print(
            f"SIMILARITY SCORE: "
            f"{result['similarity_score']:.4f}"
        )

        print("-" * 60)

        # Limit displayed text to keep terminal output readable.
        print(result["text"][:500])

        if len(result["text"]) > 500:
            print("...")

    print("\nFR-05 testing completed!")


if __name__ == "__main__":
    main()
