from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from core.vector_store import VectorStore

from core.loader.pdf import get_document_from_pdf
from core.loader.image import get_document_from_image

embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

dimensions = len(embeddings_model.embed_query("test"))
vector_store = VectorStore(embeddings_model, dimensions)
vector_store.load_or_create_faiss()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, length_function=len
)

def pdf_embed(path):
    pdf_page_list = get_document_from_pdf(path)
    vector_store.add_documents(
        text_splitter,
        pdf_page_list,
        path,
    )
    vector_store.save_faiss_instance()

def image_embed(path):
    image_page_list = get_document_from_image(path)
    vector_store.add_documents(
        text_splitter,
        image_page_list,
        path,
    )
    vector_store.save_faiss_instance()
