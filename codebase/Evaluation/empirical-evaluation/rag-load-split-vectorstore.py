from langchain.document_loaders import JSONLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import openai
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import shutil
import os


print('----Read document----')
#loader = DirectoryLoader('protocols', glob="**/*.json", loader_cls=JSONLoader, loader_kwargs={'jq_schema':'.content[] | select(.header == "Procedure") | .content'})
loader = DirectoryLoader('../AllUnityBigSplit', glob="**/*.json", loader_cls=JSONLoader, loader_kwargs={'jq_schema':'.procedures[]'})
docs = loader.load()
print('Number of documents:', len(docs))
# print(len(docs[0].page_content))
# exit()


print('----Split document----')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
docs = text_splitter.split_documents(docs) 
print('Number of document blocks:', len(docs))


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('API_BASE')


embeddings_model = OpenAIEmbeddings() # text-embedding-ada-002
persist_directory = 'docs_all_1000_100' 


if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

import time
T1 = time.time()
db = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings_model,
    persist_directory=persist_directory
)

db.persist()


print('Number of vectors:', db._collection.count())
print('Vector storage completed!')

T2=time.time()
print(f'Embedding and store take time: {T2-T1} sec.')