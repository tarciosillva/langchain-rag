from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
import os
import shutil
import nltk

# Configurar o caminho correto para o nltk_data
nltk.download("averaged_perceptron_tagger_eng")

# Carregar as variáveis de ambiente (supondo que você tenha um arquivo .env)
load_dotenv()

# Definir a chave da API do OpenAI a partir do .env
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Caminhos para os dados e a base de dados Chroma
CHROMA_PATH = "chroma"
DATA_PATH = "data/books"


def main():
    """Função principal para gerar o banco de dados de vetores."""
    generate_data_store()


def generate_data_store():
    """Carrega documentos, divide em chunks e os salva no banco de dados Chroma."""
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    """
    Carrega documentos do diretório especificado usando o DirectoryLoader.
    Certifique-se de que os arquivos estão no formato correto.
    """
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents


def split_text(documents: list):
    """
    Divide os documentos em chunks menores usando o RecursiveCharacterTextSplitter.
    
    Args:
        documents (list[Document]): Lista de documentos a serem divididos.
    
    Returns:
        list[Document]: Lista de chunks dos documentos.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Imprime detalhes de um chunk para verificação
    if len(chunks) > 10:
        document = chunks[10]
        print(document.page_content)
        print(document.metadata)

    return chunks


def save_to_chroma(chunks: list):
    """
    Salva os chunks no banco de dados Chroma.
    
    Args:
        chunks (list[Document]): Lista de chunks a serem salvos.
    """
    # Limpa o banco de dados Chroma existente, se houver.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Cria um novo banco de dados Chroma a partir dos documentos.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
