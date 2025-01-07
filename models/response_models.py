from pydantic import BaseModel
from typing import List, Optional

class QueryResponse(BaseModel):
    response: str
    sources: List[Optional[str]]
