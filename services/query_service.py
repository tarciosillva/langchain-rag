from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config.settings import Settings
from transformers import pipeline
import time

# Configurações
settings = Settings()

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

PROMPT_TEMPLATE_RESPONSE = """
Responda de forma descontraída, mantendo o respeito e valores adventistas. Seja claro e conciso:

Contexto:
{context}

Pergunta:
{question}

Resposta:
"""

import time

def process_query(query_text: str, message_context: str):
    start_time = time.time()

    summarize_start = time.time()    
    summarized_query = summarize_query(message_context)
    print(f"Tempo para resumo: {time.time() - summarize_start}s")
    
    print(f"Resumo: {summarized_query}")

    search_start = time.time()
    embedding_function = OpenAIEmbeddings(api_key=settings.openai_api_key)
    db = Chroma(persist_directory=settings.chroma_path, embedding_function=embedding_function)
    results = db.similarity_search_with_relevance_scores(summarized_query, k=3)
    print(f"Tempo para busca no banco: {time.time() - search_start}s")

    if len(results) == 0 or results[0][1] < 0.7:
        return {"response": "No matching results found.", "sources": []}

    response_start = time.time()
    context_from_db = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    full_context = f"{context_from_db}\n\n---\n\n{message_context}"
    prompt_template_response = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_RESPONSE)
    response_prompt = prompt_template_response.format_messages(context=full_context, question=query_text)
    
    model = ChatOpenAI(api_key=settings.openai_api_key, model="gpt-3.5-turbo")

    final_response = model(response_prompt)
    response_text = final_response.content.strip() if hasattr(final_response, "content") else str(final_response)
    print(f"Tempo para resposta final: {time.time() - response_start}s")

    sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]
    print(f"Tempo total: {time.time() - start_time}s")

    return {"response": response_text, "sources": sources}


def summarize_query(context: str):
    input_text = f"Context: {context}"
    
    sumary = summarizer(input_text, max_length=100, min_length=30, do_sample=False)
    
    return sumary[0]['summary_text']    