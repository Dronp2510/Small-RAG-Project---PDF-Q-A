
from ingestion.pdf_loader import extract_pdf_text
from ingestion.chunker import chunk_pages

from embeddings.embedder import Embedder
from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def main():

    print("\nTesting FR-04: Embedding Generation\n")

    # Step 1: Extract PDF text
    pages = extract_pdf_text(PDF_PATH)

    if not pages:
        print("PDF extraction failed!")
        return

    # Step 2: Generate chunks
    chunks = chunk_pages(
        pages,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    if not chunks:
        print("No chunks generated!")
        return

    print(f"Total chunks: {len(chunks)}")

    # Step 3: Initialize embedding engine
    embedder = Embedder()

    # Step 4: Generate chunk embeddings
    chunk_embeddings = embedder.embed_chunks(chunks)

    print("\nChunk embedding generation successful!")
    print(f"Number of embeddings: {len(chunk_embeddings)}")
    print(f"Embedding dimensions: {chunk_embeddings.shape}")

    # Step 5: Generate a question embedding
    question = "What is the main subject of this document?"

    question_embedding = embedder.embed_query(question)

    print("\nQuestion embedding generated successfully!")
    print(f"Question embedding shape: {question_embedding.shape}")

    # Step 6: Check normalization
    print(
        "\nFirst chunk vector norm:",
        round(float(__import__("numpy").linalg.norm(
            chunk_embeddings[0]
        )), 4)
    )

    print(
        "Question vector norm:",
        round(float(__import__("numpy").linalg.norm(
            question_embedding
        )), 4)
    )


if __name__ == "__main__":
    main()
