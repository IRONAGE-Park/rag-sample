from typing import List
from langchain_core.documents import Document
import pytesseract

from core.blip.blip import captioning

def get_document_from_image(file_path: str) -> List[Document]:
  caption = captioning(file_path)
  ocr = pytesseract.image_to_string(file_path)
  doc: List[Document] = []
  if caption:
    doc.append(Document(page_content=caption, metadata={"image_path": file_path, "type": "caption"}))
  if ocr:
    doc.append(Document(page_content=ocr, metadata={"image_path": file_path, "type": "ocr"}))
  return doc