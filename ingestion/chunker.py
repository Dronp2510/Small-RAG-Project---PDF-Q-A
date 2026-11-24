"""Page-aware character chunking with configurable overlap."""

from typing import Dict, List, Sequence


def chunk_pages(
    extracted_pages: Sequence[Dict[str, object]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, object]]:
    """Split extracted PDF pages into overlapping chunks.

    Each input page must contain ``page_number`` and ``text`` fields, as
    returned by ``extract_pdf_text``. Chunks never cross page boundaries,
    so every chunk has an unambiguous source page.

    Args:
        extracted_pages: Page records returned by the PDF loader.
        chunk_size: Maximum number of characters in each chunk.
        chunk_overlap: Number of characters repeated in the next chunk.

    Returns:
        A list of records containing chunk_id, page_number, chunk_index,
        start_char, end_char, and text.

    Raises:
        ValueError: If chunk_size or chunk_overlap is invalid.
        TypeError: If page records do not have the expected structure.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be an integer")
    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise TypeError("chunk_overlap must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[Dict[str, object]] = []

    for page in extracted_pages:
        if not isinstance(page, dict) or "page_number" not in page or "text" not in page:
            raise TypeError("Each page must be a dictionary with 'page_number' and 'text'")

        page_number = page["page_number"]
        page_text = page["text"]
        if not isinstance(page_text, str):
            raise TypeError("Page 'text' must be a string")

        page_text = page_text.strip()
        if not page_text:
            continue

        start = 0
        chunk_index = 1
        text_length = len(page_text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = page_text[start:end]

            chunks.append(
                {
                    "chunk_id": f"page_{page_number}_chunk_{chunk_index}",
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    "text": chunk_text,
                }
            )

            if end == text_length:
                break

            start = end - chunk_overlap
            chunk_index += 1

    return chunks
