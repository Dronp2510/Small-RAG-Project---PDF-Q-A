
from ingestion.pdf_loader import extract_pdf_text
from config import PDF_PATH, APP_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.chunker import chunk_pages


def main():
    print(f"\nWelcome to {APP_NAME}\n")
    print("Extracting PDF text...")

    extracted_pages = extract_pdf_text(PDF_PATH)
    if not extracted_pages:
        print("PDF extraction failed!")
        return

    print("\nPDF text extracted successfully!")
    print(f"Total pages processed: {len(extracted_pages)}")

    print("\nCreating overlapping text chunks...")
    chunks = chunk_pages(
        extracted_pages,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print("\nChunking completed successfully!")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")
    print(f"Total chunks created: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "=" * 60)
        print(f"CHUNK: {chunk['chunk_id']}")
        print(f"SOURCE PAGE: {chunk['page_number']}")
        print(f"CHARACTERS: {chunk['start_char']}–{chunk['end_char']}")
        print("=" * 60)
        print(chunk["text"])

if __name__ == "__main__":
    main()