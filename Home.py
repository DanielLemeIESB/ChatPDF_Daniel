import streamlit as st
import tempfile
import os
from pathlib import Path
from utils import cria_chain_conversa, PASTA_ARQUIVOS

# Certifique-se de que PASTA_ARQUIVOS é um objeto Path
PASTA_ARQUIVOS = Path(PASTA_ARQUIVOS)

def sidebar():
    st.sidebar.header("Home")
    uploaded_pdfs = st.file_uploader(
        'Adicione seus arquivos pdf', 
        type=['pdf'], 
        accept_multiple_files=True
    )
    
    # Lista para armazenar os caminhos dos arquivos temporários
    temp_files = []

    if uploaded_pdfs is not None:
        # Criar a pasta PASTA_ARQUIVOS se ela não existir
        if not PASTA_ARQUIVOS.exists():
            PASTA_ARQUIVOS.mkdir(parents=True, exist_ok=True)

        # Limpar arquivos PDF anteriores na pasta de armazenamento
        for arquivo in PASTA_ARQUIVOS.glob('*.pdf'):
            arquivo.unlink()
        
        # Salvar arquivos PDF temporariamente
        for pdf in uploaded_pdfs:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf.read())
                temp_file_path = temp_file.name
                temp_files.append(temp_file_path)
                # Copiar o arquivo temporário para a PASTA_ARQUIVOS
                with open(PASTA_ARQUIVOS / pdf.name, 'wb') as f:
                    f.write(open(temp_file_path, 'rb').read())
        
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
        
        # Excluir arquivos temporários após o processamento
        for temp_file_path in temp_files:
            os.remove(temp_file_path)

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
