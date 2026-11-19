
from ingestion.pdf_loader import extract_pdf_text
from config import PDF_PATH, APP_NAME


def main():

    print("\nWelcome to RAG PDF Chatbot\n")

    print("Extracting PDF text...")

    extracted_pages = extract_pdf_text(PDF_PATH)

    if not extracted_pages:
        print("PDF extraction failed!")
        return

    print("\nPDF text extracted successfully!")

    print(f"Total pages processed: {len(extracted_pages)}")

    # Display extracted text page by page
    for page in extracted_pages:

        print("\n" + "=" * 60)

        print(f"PAGE NUMBER: {page['page_number']}")

        print("=" * 60)

        if page["text"]:
            print(page["text"])
        else:
            print("[No extractable text on this page]")


if __name__ == "__main__":
    main()