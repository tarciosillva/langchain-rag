from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config.settings import Settings
from transformers import pipeline
import time

# Carregar o modelo de sumarização uma única vez
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Configurações
settings = Settings()

PROMPT_TEMPLATE_RESPONSE = """
Responda de forma descontraída, mantendo o respeito e valores adventistas. Seja claro e conciso:

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""

# Função para medir o tempo de execução de partes do código
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"Tempo total para {func.__name__}: {time.time() - start_time}s")
        return result
    return wrapper

# Função principal para processar a consulta
@measure_time
def process_query(query_text: str, message_context: str):
    # Resumo do contexto
    summarized_query = summarize_query(message_context)
    print(f"Resumo: {summarized_query}")

    # Busca no banco de dados vetorial
    embedding_function = OpenAIEmbeddings(api_key=settings.openai_api_key)
    db = Chroma(persist_directory=settings.chroma_path, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(summarized_query, k=3)

    if len(results) == 0 or results[0][1] < 0.7:
        return {"response": "No matching results found.", "sources": []}

    # Contexto final e preparação do prompt
    context_from_db = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    full_context = f"{context_from_db}\n\n---\n\n{message_context}"
    prompt_template_response = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_RESPONSE)
    response_prompt = prompt_template_response.format_messages(context=full_context, question=query_text)
    
    # Chamada ao modelo para gerar a resposta final
    model = ChatOpenAI(api_key=settings.openai_api_key, model="gpt-3.5-turbo")
    final_response = model(response_prompt)
    
    # Verificar tipo de resposta
    response_text = final_response.content.strip() if hasattr(final_response, "content") else str(final_response)

    sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]
    
    return {"response": response_text, "sources": sources}

# Função para resumir o contexto
def summarize_query(context: str):
    input_text = f"Context: {context}"
    summary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']
