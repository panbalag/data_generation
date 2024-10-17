from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from pymilvus import MilvusClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
import os


def similarity_search(query: str, k: int) -> list[Document]:
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": MILVUS_URL},
    )

    return vector_store.similarity_search(query, k=k)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

loader = DirectoryLoader('content/', glob="**/*.txt")
docs = loader.load()
#print(docs)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

milvus_client = MilvusClient(uri="./milvus_demo.db")
MILVUS_URL="./milvus_demo.db"
collection_name = "retro911"
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
docs = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

vectorstore = Milvus.from_documents(
    documents=docs,
    embedding=embeddings,
    connection_args={"uri": MILVUS_URL},
)

#print(similarity_search("Atari retro",k=1))
query = "What can you tell me about Atari Lynx?"
vectorstore.similarity_search(query, k=1)
#retriever = vectorstore.as_retriever()

template="""
[INST] 
    <<SYS>>
        You are an expert in gaming systems and games.
        You will be given a question you need to answer, and a context to provide you with information. 
        You must answer the question based as much as possible on this context.
    <</SYS>>
    Context: 
        {context}
    Question: 
        {question} 
[/INST]
"""


QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
HUGGINGFACEHUB_API_TOKEN = "XXXXXXXXXXX"
llm=HuggingFaceHub(repo_id="google/flan-t5-large", huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, model_kwargs={"temperature":1e-10})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}, retriever=vectorstore.as_retriever(), return_source_documents=False)
question = "What can you tell me about Atari Lynx?"
result = qa_chain.invoke({"query": question})
print(result)

