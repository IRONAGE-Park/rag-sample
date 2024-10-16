from typing import List
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document

def get_document_from_pdf(file_path: str) -> List[Document]:
    loader = PDFPlumberLoader(file_path)
    document_list = loader.load_and_split()
    return document_list
