from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_transformers import LongContextReorder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def query(file_path: str, query: str):

    embeddings_model = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.load_local(
        file_path, embeddings_model, allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 20})

    result = retriever.invoke(query)
    reordering = LongContextReorder()
    result = reordering.transform_documents(result)

    # for doc in result:
    #     print(doc)
    #     print("-" * 10)
    # return result

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", convert_system_message_to_human=True, temperature=0
    )

    system_prompt = (
        "You are a helpful file exploration assistant. "
        "The following is the content of the document searched based on the user's query. : {context}"
        "If the document does not contain an answer to the user's query, politely state that no matching file exists."
        "If it does contain an answer, provide a brief response to the query along with the file location​"
        "Please always answer in Korean."
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(input=query)
    print(response)


if __name__ == "__main__":
    query(
        "db/faiss",
        query="AI에서 최근 들어 사용이 더 빈번해진 기술들이 뭐야?",
    )
