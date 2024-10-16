from typing import List
from langchain_core.documents import Document
import pytesseract

from core.blip.blip import captioning

def get_document_from_image(file_path: str) -> List[Document]:
  caption = captioning(file_path)
  ocr = pytesseract.image_to_string(file_path)
  doc = [Document(page_content=f"caption: {caption}\nocr: {ocr}", metadata={ "image_path": file_path })]
  return doc