"""
Rule-based file discovery for the Argus Agent repo.

Walks the repo at run time and classifies files by agent domain so agents
always review the current file tree rather than a hardcoded list.

Classification is deterministic (extension + path rules, no LLM involved).
Files are sorted by most-recently-modified so agents prioritise active code.
Each agent is capped at MAX_FILES_PER_AGENT to keep prompt sizes sane.
"""
import logging
from pathlib import Path
from config.settings import settings

logger = logging.getLogger(__name__)

MAX_FILES_PER_AGENT = 15

# Directories to skip entirely — build artefacts, dependencies, VCS
EXCLUDED_DIRS = {
    "target", "node_modules", "build", "dist", ".git",
    "__pycache__", "venv", ".venv", ".gradle", ".mvn",
    "coverage", ".pytest_cache", ".idea", ".vscode",
}

# File name fragments that indicate test / spec files
TEST_MARKERS = ("Test.", "Spec.", ".test.", ".spec.", "_test.", "test_", "Tests.")


def _is_excluded(rel: Path) -> bool:
    """True if any path component is an excluded directory."""
    return any(part in EXCLUDED_DIRS for part in rel.parts)


def _is_test_file(path: Path) -> bool:
    return any(marker in path.name for marker in TEST_MARKERS)


def discover_files(repo_path: str | None = None) -> dict[str, list[str]]:
    """
    Walk the Argus Agent repo and return files grouped by agent domain.

    Returns a dict:  { agent_name: [relative_path, ...] }

    Paths are relative to repo_path so they can be passed directly to
    BaseAgent.read_file() which also takes relative paths.

    Returns an empty dict (not an error) if repo_path is not configured —
    agents will fall back to their hardcoded files_to_review lists.
    """
    root = Path(repo_path or settings.argus_repo_path)
    if not root.exists():
        logger.warning(f"[discovery] Repo not found at '{root}'. Agents will use fallback lists.")
        return {}

    buckets: dict[str, list[Path]] = {
        "springboot_agent":    [],
        "ml_agent":            [],
        "react_agent":         [],
        "infra_agent":         [],
        "observability_agent": [],
        "jenkins_agent":       [],
        "security_agent":      [],
        "dependency_agent":    [],
    }

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(root)
        if _is_excluded(rel):
            continue
        if _is_test_file(path):
            continue

        rel_str = str(rel)
        suffix = path.suffix.lower()
        name = path.name

        # --- springboot_agent: Java source files ---
        # security_agent also gets Java controllers/services (primary attack surface)
        if suffix == ".java":
            buckets["springboot_agent"].append(path)
            if any(kw in name for kw in ("Controller", "Service", "Security", "Config", "Auth", "Filter")):
                buckets["security_agent"].append(path)

        # --- ml_agent: Python files inside the ML service directory ---
        # security_agent also gets Python app files (Flask endpoints, subprocess usage)
        elif suffix == ".py" and "ml-service" in rel_str:
            buckets["ml_agent"].append(path)
            buckets["security_agent"].append(path)

        # --- react_agent: JSX/TSX anywhere + JS files inside the React frontend ---
        # security_agent gets JS service files (token storage, API calls)
        elif suffix in (".jsx", ".tsx"):
            buckets["react_agent"].append(path)
        elif suffix == ".js" and "react-frontend" in rel_str:
            buckets["react_agent"].append(path)
            if any(kw in name.lower() for kw in ("service", "auth", "api", "token")):
                buckets["security_agent"].append(path)

        # --- jenkins_agent: Jenkinsfile (primary) + docker-compose (service context) ---
        # security_agent also gets Jenkinsfile (credentials handling in CI)
        elif name == "Jenkinsfile":
            buckets["jenkins_agent"].append(path)
            buckets["security_agent"].append(path)

        # --- infra_agent: Docker Compose, Nginx configs ---
        # docker-compose also feeds security_agent (ports) and dependency_agent (image versions)
        elif "docker-compose" in name and suffix in (".yml", ".yaml"):
            buckets["jenkins_agent"].append(path)
            buckets["infra_agent"].append(path)
            buckets["security_agent"].append(path)
            buckets["dependency_agent"].append(path)
        elif suffix in (".conf", ".nginx") and "nginx" in rel_str.lower():
            buckets["infra_agent"].append(path)
            buckets["security_agent"].append(path)

        # --- dependency_agent: Maven, pip, npm manifests, and Dockerfiles ---
        elif name == "pom.xml":
            buckets["dependency_agent"].append(path)
        elif name == "requirements.txt":
            buckets["dependency_agent"].append(path)
        elif name == "package.json" and "node_modules" not in rel_str:
            buckets["dependency_agent"].append(path)
        elif name == "Dockerfile":
            buckets["dependency_agent"].append(path)

        # --- observability_agent: Prometheus, Alertmanager, Grafana YAML ---
        elif suffix in (".yml", ".yaml") and any(
            kw in name.lower()
            for kw in ("prometheus", "alert", "alertmanager", "grafana", "datasource", "dashboard")
        ):
            buckets["observability_agent"].append(path)

    # Sort each bucket by mtime desc (most recently touched first), then cap
    discovered: dict[str, list[str]] = {}
    for agent, paths in buckets.items():
        sorted_paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
        capped = sorted_paths[:MAX_FILES_PER_AGENT]
        discovered[agent] = [str(p.relative_to(root)) for p in capped]
        if discovered[agent]:
            logger.info(f"[discovery] {agent}: {len(discovered[agent])} file(s)")
        else:
            logger.info(f"[discovery] {agent}: no files found")

    return discovered


def summarise_discovery(discovered: dict[str, list[str]]) -> str:
    """
    One-line-per-agent summary for injecting into LLM prompts.
    Example:
      springboot_agent: 3 file(s) — JenkinsService.java, JenkinsController.java, ...
    """
    lines = []
    for agent, files in discovered.items():
        if files:
            names = ", ".join(Path(f).name for f in files[:5])
            suffix = f" (+ {len(files) - 5} more)" if len(files) > 5 else ""
            lines.append(f"  {agent}: {len(files)} file(s) — {names}{suffix}")
        else:
            lines.append(f"  {agent}: no files discovered (will use fallback list)")
    return "\n".join(lines)
