from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str = ""
    orchestrator_model: str = "claude-sonnet-4-6"
    agent_model: str = "claude-haiku-4-5-20251001"

    # LangSmith
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = True
    langchain_project: str = "argus-cortex"

    # GitHub
    github_token: str = ""
    github_org: str = "ASREDDY03"
    argus_repo: str = "argus-agent"

    # Local path to Argus Agent
    argus_repo_path: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
