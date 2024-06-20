import os
import json
import gdown
from pathlib import Path
import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

from configs import *

# URL do arquivo JSON no Google Drive
gdrive_url = 'https://drive.google.com/uc?id=1oTfj9BCKxPtowyRHADsdHagQBfGsiisb'

# Caminho temporário para salvar o arquivo JSON
temp_json_path = '/tmp/api_key.json'

# Baixar o arquivo JSON do Google Drive
gdown.download(gdrive_url, temp_json_path, quiet=False)

# Ler a chave da API do arquivo JSON
with open(temp_json_path, 'r') as f:
    data = json.load(f)
    api_key = data['OPENAI_API_KEY']

# Definir a variável de ambiente com a chave da API
os.environ['OPENAI_API_KEY'] = api_key

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

PASTA_ARQUIVOS = Path(__file__).parent / 'arquivos'

def importacao_documentos():
    documentos = []
    for arquivo in PASTA_ARQUIVOS.glob('*.pdf'):
        loader = PyPDFLoader(str(arquivo))
        documentos_arquivo = loader.load()
        documentos.extend(documentos_arquivo)
    return documentos

def split_de_documentos(documentos):
    recur_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
        separators=["/n\n", "\n", ".", " ", ""]
    )
    documentos = recur_splitter.split_documents(documentos)

    for i, doc in enumerate(documentos):
        doc.metadata['source'] = doc.metadata['source'].split('/')[-1]
        doc.metadata['doc_id'] = i
    return documentos

def cria_vector_store(documentos):
    embedding_model = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(
        documents=documentos,
        embedding=embedding_model
    )
    return vector_store

def cria_chain_conversa():

    documentos = importacao_documentos()
    documentos = split_de_documentos(documentos)
    vector_store = cria_vector_store(documentos)

    chat = ChatOpenAI(model=get_config('model_name'))
    memory = ConversationBufferMemory(
        return_messages=True,
        memory_key='chat_history',
        output_key='answer'
        )
    retriever = vector_store.as_retriever(
        search_type=get_config('retrieval_search_type'),
        search_kwargs=get_config('retrieval_kwargs')
    )
    prompt = PromptTemplate.from_template(get_config('prompt'))
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

    st.session_state['chain'] = chat_chain

def main():
    with st.sidebar:
        sidebar()
    chat_window()

if __name__ == '__main__':
    main()
