from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config.settings import Settings

# Configurações
settings = Settings()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def process_query(query_text: str):
    # Inicializar o banco de dados
    embedding_function = OpenAIEmbeddings(api_key=settings.openai_api_key)
    db = Chroma(persist_directory=settings.chroma_path, embedding_function=embedding_function)

    # Procurar na base de dados
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        return {"response": "No matching results found.", "sources": []}

    # Criar contexto
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format_messages(context=context_text, question=query_text)

    # Chamar o modelo
    model = ChatOpenAI(api_key=settings.openai_api_key, model="gpt-3.5-turbo")
    response = model(prompt)

    # Extrair texto da resposta
    response_text = response.content if hasattr(response, "content") else str(response)

    # Extrair fontes
    sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]

    return {"response": response_text, "sources": sources}
