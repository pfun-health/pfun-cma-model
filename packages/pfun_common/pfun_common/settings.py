from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings
    """
    
    debug: bool = False
    server_scheme: str = "http"
    server_host: str = "localhost"
    server_port: str = "8001"
    gradio_server_scheme: str = "http"
    gradio_server_host: str = "localhost"
    gradio_server_port: str = "7860"
    perplexity_api_key: str = ""
    secret_key: str = "SChp11HMytLzSj3gaJQAJhq5sqc9Aicnz"
    google_cloud_project_id: str = "pfun-cma-model"
    google_cloud_location: str = "us-central1"

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=('.env', ),
        env_file_encoding='utf-8',
        extra='allow'
    )
    
    @property
    def llm_gen_scenario_endpoint(self) -> str:
        """
        LLM generate-scenario endpoint URL.
        
        :param self: Description
        :return: Description
        :rtype: str
        """
        return f"{self.server_scheme}://{self.server_host}:{self.server_port}/llm/generate-scenario"



def get_settings() -> Settings:
    """Initialize the settings object (dependency injection helper method)."""
    return Settings()