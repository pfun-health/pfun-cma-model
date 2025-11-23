from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings
    """
    
    DEBUG: bool = False
    GOOGLE_CLOUD_PROJECT_ID: str = "pfun-cma-model"
    GOOGLE_CLOUD_LOCATION: str = "us-central1"
    
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')



def get_settings():
    return Settings()