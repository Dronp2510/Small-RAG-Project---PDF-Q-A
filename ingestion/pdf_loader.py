
from pathlib import Path
import pymupdf as fitz  # PyMuPDF

def extract_pdf_text(pdf_path):
    """
    Extract text from a PDF page by page.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: A list of dictionaries containing page numbers and text.
    """

    extracted_pages = []

    try:
        pdf_document = fitz.open(pdf_path)

        for page_index, page in enumerate(pdf_document):

            # Extract text from the current page
            page_text = page.get_text("text")

            # Store page number and extracted text
            page_data = {
                "page_number": page_index + 1,
                "text": page_text.strip()
            }

            extracted_pages.append(page_data)

        pdf_document.close()

        return extracted_pages

    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return []
