
from ingestion.pdf_loader import extract_pdf_text
from ingestion.chunker import chunk_pages

from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from generation.llm_client import GeminiClient

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

    # STEP 6: Initialize Gemini for answer generation
    print("\nStep 6: Initializing Gemini...")
    try:
        gemini = GeminiClient()
    except Exception as error:
        print(f"Gemini initialization failed: {error}")
        return

    print("Gemini initialized successfully!")

    # STEP 7: Accept a question and retrieve relevant chunks
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

        # Generate an answer using only the retrieved PDF context
        print("\nGenerating an answer from the PDF context...")
        try:
            answer = gemini.generate_answer(
                question=question,
                retrieved_chunks=retrieved_chunks,
            )
        except Exception as error:
            print(f"\nAnswer generation failed: {error}")
            print("Check Gemini API access and try again.")
            continue

        # Collect unique source pages in retrieval order
        source_pages = []
        for result in retrieved_chunks:
            page_number = result.get("page_number")
            if page_number is not None and page_number not in source_pages:
                source_pages.append(page_number)

        print("\nAnswer")
        print("=" * 60)
        print(answer)

        if source_pages:
            print("\nSource pages: " + ", ".join(map(str, source_pages)))

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
