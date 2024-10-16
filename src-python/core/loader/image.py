from typing import List
from langchain_core.documents import Document

from core.blip.blip import captioning

def get_document_from_image(file_path: str) -> List[Document]:
  caption = captioning(file_path)
  doc = [Document(page_content=caption, metadata={ "image_path": file_path })]
  return doc