from fastapi import FastAPI
from api.routes import router
from config.settings import Settings

# Instância do FastAPI
app = FastAPI()

# Carregar configurações
settings = Settings()

# Incluir rotas
app.include_router(router)

@app.get("/")
def health_check():
    return {"message": "API is running!"}
