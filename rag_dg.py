from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from pymilvus import MilvusClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, LlamaTokenizer, LlamaForCausalLM
import torch
from langchain_huggingface import HuggingFacePipeline
import os
import re
from nltk import sent_tokenize

def clean_output(answer):
   sentences = sent_tokenize(answer)
   seen = set()
   unique_sentences = [s for s in sentences if not (s in seen or seen.add(s))]
   cleaned_answer = ' '.join(unique_sentences)
   return cleaned_answer


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

template = """
You are an expert on gaming systems and games.
You will be given a question and some context to help you answer it.
Please provide an accurate and comprehensive response based on the provided context.
Context: 
{context}
Question: 
{question}
"""

#Download the model locally
#model_name = "meta-llama/Llama-2-7b-hf" 
model_name="meta-llama/Llama-2-13b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer,temperature=1e-10,max_new_tokens=2000)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}, retriever=vectorstore.as_retriever(), return_source_documents=False)

games=["Mario Kart VR", "Tetris VR", "Grant's escape adventure", "Retro911 to the rescue", "AmigaTales", "Frank and the wombats", "Sir Cedric of the East Village Sim", "Taylor's Cat hoarding sim", "Legare's university 2k25", "Street Taco Bonanza", "Flipping Waffles", "Best Chronicles", "Cyberduck", "Escape from Broadcom", "Ridge rager", "Super Mario Bros",   "The Legend of Zelda","Pac-Man","Sonic the Hedgehog","Tetris","Street Fighter II","Donkey Kong",    "Final Fantasy VII","Space Invaders","Castlevania","Mega Man 2",    "Doom","Galaga",        "Metroid",  "Contra"]

gaming_systems= ["Cat9600","Intellico",   "Switchblade",  "GraniteTastic","iGameLab","16bitClaw","TheClaw","pipinstaller",    "genConsole",   "Force5",   "NinjaX",   "Neo X","Magneto OdX",  "TinTin 2000",  "DynoVision", "Atari 2600","Nintendo Entertainment System",   "Sega Genesis","Super Nintendo Entertainment System",   "Sega Saturn",  "Sony PlayStation", "Nintendo 64",  "Sega Dreamcast",   "Game Boy", "Commodore 64","TurboGrafx-16", "Atari Lynx",   "Panasonic 3D0",    "TurboGrafx-16",    "Sega Saturn"]

for game in games:
  output_file = "content/cheat_sheet/" + game +".txt"
  print("Generating cheat sheet for the game " + game + " .... IN PROGRESS")
  question = "Give me a comprehensive cheat sheet including key points, strategies, important items, tips for quick reference, for the game " + game
  # Question to use for generating character guide
  # question = "Give me a comprehensive character guide including character's backstory, abilities, strengths and weaknesses for the game " + game
  result = qa_chain.invoke({"query": question})
  text = str(result['result'])
  print("Writing output to file " +  output_file)
  with open(output_file, "w") as f:
      f.write(text)
  print("Generating cheat sheet for the game " + game + " .... COMPLETE!")
