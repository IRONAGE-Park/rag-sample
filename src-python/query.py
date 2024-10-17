from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.embeddings import OpenVINOBgeEmbeddings
from langchain_community.document_transformers import LongContextReorder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.retrievers.multi_query import LineListOutputParser, MultiQueryRetriever

from operator import itemgetter

from dotenv import load_dotenv
import os

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def query(file_path: str, query: str):
    embeddings_model = OpenVINOBgeEmbeddings(
        model_name_or_path="D:\\Intel\\ov_bge-m3",
        model_kwargs={"device": "NPU", "compile": False},
        encode_kwargs={
            "mean_pooling": False,
            "normalize_embeddings": True,
            "batch_size": 1,
        },
    )
    embeddings_model.ov_model.reshape(1, 512)
    embeddings_model.ov_model.compile()

    vector_store = FAISS.load_local(
        file_path, embeddings_model, allow_dangerous_deserialization=True
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    multi_query_prompt = PromptTemplate.from_template(
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from a vector
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search.
        Provide these alternative questions separated by newlines.
        Keep in mind that understanding the characteristics of the vector database and extracting key phrases from sentence-based queries is essential.
        Please Do not include any other text than the list of questions.

        #ORIGINAL QUESTION:
        {question}

        #Answer in Korean:
        """
    )
    multi_query_chain = (
        {
            "question": itemgetter("question"),
        }
        | multi_query_prompt
        | llm
        | LineListOutputParser()
    )
    multi_query_retriever = MultiQueryRetriever(
        llm_chain=multi_query_chain,
        parser_key="lines",
        retriever=vector_store.as_retriever(search_kwargs={"k": 20}),
    )

    system_prompt = (
        "You are a helpful file exploration assistant."
        "Your mission is to check whether the requested information is included in the given document,"
        "or if the content can be inferred based on the provided document, and to provide an answer to the user’s question."
        "If the document does not contain relevant information or if the content cannot be inferred, politely state that no matching file exists."
        "If it does contain an answer, or if it can be inferred by the provided document, provide a brief response to the query along with the file location​"
        "Do not fabricate or invent any information that is not provided in the document."
        "Provided source is path of the file."
        "Please always answer in Korean."
        "The following is the content of the document searched based on the user's query. : {context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        messages=[
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    chain = (
        {
            "context": multi_query_retriever | RunnableLambda(reorder_documents),
            "question": RunnablePassthrough(),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(input=query)
    response = chain.astream(input=query)


    print("Answer : ", response)

def reorder_documents(docs):
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(docs)
    return reordered_docs

if __name__ == "__main__":
    query(
        "db/faiss",
        query="AI에서 최근 들어 사용이 더 빈번해진 기술들이 뭐야?",
    )
