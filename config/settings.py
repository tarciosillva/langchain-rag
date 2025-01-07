from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    chroma_path: str = "chroma"
    
    class Config:
        env_file = ".env"
