from pathlib import Path

import fitz
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from PIL import Image
import pytesseract


MIN_TEXT_CHARS = 40
OCR_ZOOM = 2
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_document(path: str | Path):
    file_path = Path(path)
    if file_path.suffix.lower() in IMAGE_EXTENSIONS:
        return load_image(file_path)

    return load_pdf(file_path)


def load_pdf(path: str | Path):
    file_path = Path(path)
    file_name = file_path.name
    loader = PyPDFLoader(str(path))
    docs = loader.load()

    extracted_docs = _add_ocr_for_low_text_pages(file_path, docs)

    for doc in extracted_docs:
        doc.metadata["source"] = file_name

    readable_docs = [doc for doc in extracted_docs if doc.page_content.strip()]
    if not readable_docs:
        raise ValueError("No readable text found in this PDF, even after OCR.")

    return readable_docs


def load_image(path: str | Path):
    file_path = Path(path)
    with Image.open(file_path) as image:
        image = image.convert("RGB")
        text = pytesseract.image_to_string(image, config="--psm 6").strip()

    if not text:
        raise ValueError("No readable text found in this image after OCR.")

    return [
        Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "page": 0,
                "extraction": "ocr",
            },
        )
    ]


def _add_ocr_for_low_text_pages(path: Path, docs: list[Document]) -> list[Document]:
    pdf = fitz.open(path)
    try:
        docs_by_page = {
            doc.metadata.get("page"): doc
            for doc in docs
            if isinstance(doc.metadata.get("page"), int)
        }
        output = []

        for page_index in range(pdf.page_count):
            base_doc = docs_by_page.get(page_index)
            existing_text = base_doc.page_content.strip() if base_doc else ""

            if len(existing_text) >= MIN_TEXT_CHARS:
                output.append(base_doc)
                continue

            ocr_text = _ocr_page(pdf[page_index]).strip()
            page_text = ocr_text or existing_text
            output.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": path.name,
                        "page": page_index,
                        "extraction": "ocr" if ocr_text else "text",
                    },
                )
            )

        return output
    finally:
        pdf.close()


def _ocr_page(page) -> str:
    matrix = fitz.Matrix(OCR_ZOOM, OCR_ZOOM)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )

    return pytesseract.image_to_string(image, config="--psm 6")
