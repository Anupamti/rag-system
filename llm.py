import getpass
from dotenv import load_dotenv
import os

load_dotenv()  

api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = api_key

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_core.messages import HumanMessage, AIMessage

import os

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

def load_documents(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        documents.extend(loader.load())
    return documents


folder_path = "docs"
documents = load_documents(folder_path)
# print(f"Loaded {len(documents)} documents from the folder.")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

splits = text_splitter.split_documents(documents)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
document_embeddings = embedding_function.embed_documents([split.page_content for split in splits])
# print(document_embeddings[0][:5])  # Printing first 5 elements of the first embedding




collection_name = "my_collection"
vectorstore = Chroma.from_documents(
    collection_name=collection_name,
    documents=splits,
    embedding=embedding_function,
    persist_directory="./chroma_db"
)
print("Vector store created and persisted to './chroma_db'")

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
# retriever_results = retriever.invoke("When was GreenGrow Innovations founded?")
# print(retriever_results)




template = """Answer the question based only on the following context:
{context}
Question: {question}
Answer: """

prompt = ChatPromptTemplate.from_template(template)

def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | docs2str, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# question = "What is the Acme When was Acme Corp founded?"
# response = rag_chain.invoke(question)
# print(f"Question: {question}")
# print(f"Answer: {response}")

while(True):
    query = input()
    if query == 'exit':
        break
    else:
        response = rag_chain.invoke(query)
        print(f"Question: {query}")
        print(f"Answer: {response}")