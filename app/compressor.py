from pathlib import Path

from PIL import Image
from pypdf import PdfReader, PdfWriter


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS


def compress_upload(input_path: Path, output_dir: Path) -> Path:
    suffix = input_path.suffix.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    if suffix in IMAGE_EXTENSIONS:
        return _compress_image(input_path, output_dir)

    if suffix in PDF_EXTENSIONS:
        return _compress_pdf(input_path, output_dir)

    raise ValueError("Only PDF, PNG, JPG, JPEG, and WEBP files are supported.")


def _compress_image(input_path: Path, output_dir: Path) -> Path:
    output_path = output_dir / f"{input_path.stem}.webp"

    with Image.open(input_path) as image:
        image = image.convert("RGB")
        image.thumbnail((2200, 2200))
        image.save(output_path, "WEBP", quality=78, method=6)

    return output_path


def _compress_pdf(input_path: Path, output_dir: Path) -> Path:
    output_path = output_dir / input_path.name
    reader = PdfReader(str(input_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    for page in writer.pages:
        page.compress_content_streams()
    with output_path.open("wb") as file:
        writer.write(file)
    return output_path
