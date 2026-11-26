
from ingestion.pdf_loader import extract_pdf_text
from ingestion.chunker import chunk_pages

from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore

from config import (
    PDF_PATH,
    APP_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def main():
    print(f"\nWelcome to {APP_NAME}\n")

    # STEP 1: Extract text from the PDF
    print("Step 1: Extracting PDF text...")

    extracted_pages = extract_pdf_text(PDF_PATH)

    if not extracted_pages:
        print("PDF extraction failed or the PDF contains no pages.")
        return

    print("PDF text extracted successfully!")
    print(f"Total pages processed: {len(extracted_pages)}")

    # STEP 2: Split the extracted text into chunks
    print("\nStep 2: Creating text chunks...")

    chunks = chunk_pages(
        extracted_pages,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        print("No text chunks were generated from the PDF.")
        return

    print("Chunking completed successfully!")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")
    print(f"Total chunks created: {len(chunks)}")

    # STEP 3: Generate embeddings for all chunks
    print("\nStep 3: Generating document embeddings...")

    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)

    print("Document embeddings generated successfully!")
    print(f"Embedding shape: {embeddings.shape}")

    # STEP 4: Create the FAISS vector store
    print("\nStep 4: Initializing FAISS vector store...")

    vector_store = VectorStore(
        embedding_dimension=embedder.embedding_dimension
    )

    # STEP 5: Add embeddings and their corresponding chunks to FAISS
    print("\nStep 5: Indexing document embeddings...")

    vector_store.add_embeddings(
        embeddings=embeddings,
        chunks=chunks,
    )

    print("\nDocument indexing completed!")
    print(f"Total vectors indexed: {vector_store.get_index_size()}")

    # STEP 6: Accept a question and retrieve relevant chunks
    print("\n" + "=" * 60)
    print("PDF QUESTION RETRIEVAL")
    print("=" * 60)

    while True:
        question = input(
            "\nEnter your question (or type 'exit' to quit): "
        ).strip()

        if question.lower() in ("exit", "quit"):
            print("\nExiting the application. Goodbye!")
            break

        if not question:
            print("Please enter a valid question.")
            continue

        # Generate an embedding for the user's question
        query_embedding = embedder.embed_query(question)

        # Retrieve the five most relevant chunks
        retrieved_chunks = vector_store.search(
            query_embedding=query_embedding,
            top_k=5,
        )

        if not retrieved_chunks:
            print("No relevant chunks were found.")
            continue

        # Display retrieved chunks and their metadata
        print("\nTop Retrieved Chunks")
        print("=" * 60)

        for rank, result in enumerate(retrieved_chunks, start=1):
            print(f"\nRANK: {rank}")
            print(f"CHUNK ID: {result['chunk_id']}")
            print(f"SOURCE PAGE: {result['page_number']}")
            print(f"SIMILARITY SCORE: {result['similarity_score']:.4f}")
            print("-" * 60)

            print(result["text"])

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
