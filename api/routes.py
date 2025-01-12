from fastapi import APIRouter, HTTPException
from models.request_models import QueryRequest
from models.response_models import QueryResponse
from services.query_service import process_query

# Inicializar o roteador
router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    response = process_query(request.query_text, request.message_context)
    if not response["response"]:
        raise HTTPException(status_code=404, detail="No matching results found.")
    return response
