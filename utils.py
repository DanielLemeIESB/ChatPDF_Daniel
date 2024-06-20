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
import gdown
import json
import os
import sys
import requests  # Adicionando a importação do requests
from dotenv import load_dotenv, find_dotenv

from configs import *

# URL do arquivo JSON no Google Drive
gdrive_url = 'https://drive.google.com/uc?id=1HNepMO6p9uWXVywiBrX5dQdy0vJQcVn_'

# Caminho temporário para salvar o arquivo JSON
temp_json_path = '/tmp/config_api_keys.json'

# Função para suprimir a saída padrão
class SuppressStdoutStderr:
    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self.stdout
        sys.stderr = self.stderr

# Verificação de URL antes de tentar o download
response = requests.get(gdrive_url)
if response.status_code == 200:
    # Baixar o arquivo JSON do Google Drive suprimindo a saída padrão
    with SuppressStdoutStderr():
        gdown.download(gdrive_url, temp_json_path, quiet=True)

    # Verificar se o arquivo foi baixado e se ele é JSON válido
    try:
        with open(temp_json_path, 'r') as f:
            data = json.load(f)
            api_key = data['openai_api_key']
    except json.JSONDecodeError:
        st.error("O arquivo baixado não é um JSON válido. Verifique a URL do Google Drive e o conteúdo do arquivo.")
        st.stop()
    except FileNotFoundError:
        st.error("O arquivo JSON não foi encontrado. Verifique se o download foi bem-sucedido.")
        st.stop()

    # Definir a variável de ambiente com a chave da API
    os.environ['OPENAI_API_KEY'] = api_key

    # Carregar variáveis de ambiente do arquivo .env
    _ = load_dotenv(find_dotenv())
else:
    st.error("Não foi possível acessar a URL do Google Drive. Verifique se a URL está correta e acessível.")
    st.stop()

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
