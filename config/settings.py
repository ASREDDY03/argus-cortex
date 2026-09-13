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

    # Webhook server
    webhook_secret: str = ""   # GitHub webhook secret (set when registering the webhook)
    webhook_port: int = 8000

    # Web dashboard
    dashboard_port: int = 8001

    # Scheduled runs (APScheduler cron — leave empty to disable)
    # Examples:  "0 9 * * 1"    every Monday at 09:00
    #            "0 0 * * *"    every day at midnight
    #            "0 */6 * * *"  every 6 hours
    schedule_cron: str = ""
    schedule_goal: str = "Scheduled full audit — security, CI/CD, code quality, and dependency health"

    # Slack notifications (Incoming Webhook URL — set in .env to enable)
    slack_webhook_url: str = ""

    # CI validation before opening PR
    ci_validation: bool = True   # set to false to skip CI check and open PR immediately
    ci_timeout: int = 600        # seconds to wait for CI to complete (default 10 min)

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
