import os
from typing import List
from faiss import IndexFlatL2
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStore:
    DB_FOLDER_PATH_NAME = "db"
    FAISS_FOLDER_PATH_NAME = "faiss"
    INDEX_FOLDER_PATH_NAME = "faiss_file_index"

    INDEX_FILE_NAME = "index.json"
    DB_FILE_NAME = "index.faiss"

    INDEX_FOLDER_PATH = os.path.join(
        DB_FOLDER_PATH_NAME, FAISS_FOLDER_PATH_NAME, INDEX_FOLDER_PATH_NAME
    )
    INDEX_FILE_PATH = os.path.join(INDEX_FOLDER_PATH, INDEX_FILE_NAME)
    DB_FOLDER_PATH = os.path.join(DB_FOLDER_PATH_NAME, FAISS_FOLDER_PATH_NAME)
    DB_FILE_PATH = os.path.join(DB_FOLDER_PATH, DB_FILE_NAME)

    def __init__(self, embeddings_model: HuggingFaceEmbeddings, dimensions: int):
        self.embeddings_model = embeddings_model
        self.dimensions = dimensions
        self.faiss_instance: FAISS | None = None

    def save_index_file(self, file_path: str, index: List[str]):
        try:
            import json
        except ImportError:
            raise ImportError("json module not found")
        INDEX_OBJ = {
            "file_path": file_path,
            "index": index,
        }

        if not os.path.exists(self.INDEX_FOLDER_PATH):
            os.makedirs(self.INDEX_FOLDER_PATH)

        with open(self.INDEX_FILE_PATH, mode="a+") as f:
            json.dump(INDEX_OBJ, f)

    def load_or_create_faiss(self):
        try:
            if os.path.exists(self.DB_FILE_PATH):
                self.faiss_instance = FAISS.load_local(
                    folder_path=self.DB_FOLDER_PATH,
                    embeddings=self.embeddings_model,
                    allow_dangerous_deserialization=True,
                )
            else:
                print("Creating new FAISS instance")
                self.faiss_instance = FAISS(
                    embedding_function=self.embeddings_model,
                    index=IndexFlatL2(self.dimensions),
                    docstore=InMemoryDocstore(),
                    distance_strategy=DistanceStrategy.COSINE,
                    index_to_docstore_id={},
                )
        except Exception as e:
            raise Exception(f"Error loading or creating FAISS instance: {e}")

    def get_faiss_instance(self) -> FAISS:
        if self.faiss_instance is None:
            raise ValueError(
                "FAISS instance is not loaded or created yet. Call `load_or_create_faiss()` first."
            )
        return self.faiss_instance

    def save_faiss_instance(self):
        if self.faiss_instance is None:
            raise ValueError(
                "FAISS instance is not loaded or created yet. Call `load_or_create_faiss()` first."
            )

        if not os.path.exists(self.DB_FOLDER_PATH):
            os.makedirs(self.DB_FOLDER_PATH)

        self.faiss_instance.save_local(self.DB_FOLDER_PATH)

    def add_documents(
        self,
        text_splitter,
        documents,
        document_file_path,
    ):
        if self.faiss_instance is None:
            raise ValueError(
                "FAISS instance is not loaded or created yet. Call `load_or_create_faiss()` first."
            )

        documents = text_splitter.split_documents(documents)
        added_documents_index = self.faiss_instance.add_documents(documents)

        self.save_index_file(document_file_path, added_documents_index)
