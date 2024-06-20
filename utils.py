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
from dotenv import load_dotenv, find_dotenv

from configs import *

# URL do arquivo JSON no Google Drive
gdrive_url = 'https://drive.google.com/file/d/1oTfj9BCKxPtowyRHADsdHagQBfGsiisb/view?usp=drive_link'

# Caminho temporário para salvar o arquivo JSON
temp_json_path = '/tmp/api_key.json'

# Baixar o arquivo JSON do Google Drive
gdown.download(gdrive_url, temp_json_path, quiet=True)

# Verificar se o arquivo foi baixado e se ele é JSON válido
try:
    with open(temp_json_path, 'r') as f:
        data = json.load(f)
        api_key = data['OPENAI_API_KEY']
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

def sidebar():
    uploaded_pdfs = st.file_uploader(
        'Adicione seus arquivos pdf', 
        type=['pdf'], 
        accept_multiple_files=True
    )
    
    if uploaded_pdfs is not None:
        # Limpar arquivos PDF anteriores na pasta de armazenamento
        for arquivo in PASTA_ARQUIVOS.glob('*.pdf'):
            arquivo.unlink()
        
        # Salvar arquivos PDF carregados
        for pdf in uploaded_pdfs:
            with open(PASTA_ARQUIVOS / pdf.name, 'wb') as f:
                f.write(pdf.read())
    
    label_botao = 'Inicializar ChatBot'
    if 'chain' in st.session_state:
        label_botao = 'Atualizar ChatBot'
    if st.button(label_botao, use_container_width=True):
        if len(list(PASTA_ARQUIVOS.glob('*.pdf'))) == 0:
            st.error('Adicione arquivos .pdf para inicializar o chatbot')
        else:
            st.success('Inicializando o ChatBot...')
            cria_chain_conversa()
            st.rerun()

def chat_window():
    st.header('🤖 Bem-vindo ao Chat com PDFs do Daniel', divider=True)

    if 'chain' not in st.session_state:
        st.error('Faça o upload de PDFs para começar!')
        st.stop()
    
    chain = st.session_state['chain']
    memory = chain.memory

    mensagens = memory.load_memory_variables({})['chat_history']

    container = st.container()
    for mensagem in mensagens:
        chat = container.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    nova_mensagem = st.chat_input('Converse com seus documentos...')
    if nova_mensagem:
        chat = container.chat_message('human')
        chat.markdown(nova_mensagem)
        chat = container.chat_message('ai')
        chat.markdown('Gerando resposta')

        resposta = chain.invoke({'question': nova_mensagem})
        st.session_state['ultima_resposta'] = resposta
        st.rerun()

def main():
    with st.sidebar:
        sidebar()
    chat_window()

if __name__ == '__main__':
    main()
