"""
LangSmith observability setup.

Call init_tracing() once at startup in main.py.
LangGraph traces are sent automatically.
Individual functions decorated with @traceable appear as nested spans.
"""
import os
from config.settings import settings


_PLACEHOLDER_KEYS = {"your_langsmith_api_key_here", "your-langsmith-api-key", ""}


def init_tracing():
    """
    Set LangSmith env vars from settings and enable tracing.
    Must be called before any LangGraph or LangChain code runs.
    Only activates if the key looks like a real key (not a placeholder).
    """
    key = settings.langchain_api_key or ""
    if not key or key in _PLACEHOLDER_KEYS or not key.startswith("ls__"):
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    os.environ["LANGCHAIN_API_KEY"] = key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    return True


def get_trace_url(run_id: str) -> str | None:
    """
    Return LangSmith trace URL for a given run_id.
    Only works if tracing is enabled.
    """
    if not settings.langchain_api_key:
        return None
    project = settings.langchain_project
    return f"https://smith.langchain.com/o/projects/{project}/runs/{run_id}"
