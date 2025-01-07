from pydantic import BaseModel

class QueryRequest(BaseModel):
    query_text: str
