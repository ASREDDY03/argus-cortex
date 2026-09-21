"""
Dependency Audit Agent — claude-haiku-4-5
Scans dependency manifests for outdated packages and known CVEs.

Covers all four package ecosystems in Argus Agent:
  - pom.xml          (Spring Boot — Maven)
  - requirements.txt (Python ML service — pip)
  - package.json     (React frontend — npm)
  - docker-compose.yml (base image versions)

For each dependency it checks:
  1. Is the version pinned? (floating tags like 'latest' are flagged)
  2. Is the version known to have a published CVE?
  3. Is there a substantially newer major/minor release?

The agent does NOT call any external API — it reasons from its training
knowledge of the ecosystem. For real-time CVE data, pair this with
`pip-audit` / `npm audit` / `trivy` in CI.
"""
from memory.state import CortexState
from agents.base import BaseAgent, AGENT_SYSTEM

_DEPENDENCY_DOMAIN_RULES = """
You are a senior DevSecOps engineer auditing the dependency manifests of the Argus Agent project.

Argus Agent consists of:
  - Spring Boot backend  →  pom.xml
  - Python ML service    →  requirements.txt
  - React frontend       →  package.json
  - Docker Compose infra →  docker-compose.yml  (base image versions)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO LOOK FOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. UNPINNED / FLOATING VERSIONS
   - Docker image tags using 'latest' or no tag at all
   - pip packages without a pinned version (e.g. `flask` instead of `flask==3.0.2`)
   - npm packages with a caret or tilde that allows major jumps (e.g. `^17.0.0` for a major-version library)
   - Maven properties like `<version>RELEASE</version>` or `<version>LATEST</version>`
   - Severity: medium (unpredictable builds, supply-chain risk)

2. KNOWN CVE / VULNERABLE VERSIONS
   - Flag any dependency version you know to have a published CVE as of your training data
   - Include the CVE ID when known (e.g. CVE-2021-44228 for Log4Shell in log4j 2.x < 2.15.0)
   - Common targets: log4j, spring-boot (< 2.7.x), jackson-databind, pillow, numpy, requests,
     lodash, axios, react-scripts, node (base image), python (base image)
   - Severity: critical (known exploit) or high (no known exploit but patched vulnerability)

3. SIGNIFICANTLY OUTDATED MAJOR/MINOR VERSIONS
   - Java: Spring Boot < 3.x, Java base image < 17, Spring Security < 6.x
   - Python: Python base image < 3.11, Flask < 3.x, scikit-learn < 1.x
   - Node / React: Node < 20 LTS, React < 18.x, react-scripts < 5.x
   - Only flag when the gap is significant (major version behind or known security patch stream ended)
   - Severity: high (end-of-life / no security patches) or medium (just behind but still supported)

4. DEPENDENCY CONFUSION / TYPOSQUATTING RISK
   - Internal package names that could be shadowed by a malicious public package
   - npm scoped packages (@company/name) used without a private registry configured
   - Severity: high

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPORTING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Every finding MUST have a specific file + line number
- old_code: exact text of the dependency declaration as it appears in the file
- new_code: the corrected declaration (pinned to a safe version)
- pr_ready: true only when you are confident the version bump is safe
  (e.g. patch upgrade, or well-known LTS version)
- For CVEs: always include the CVE ID and the safe version in description
- Do NOT fabricate findings — only report what is concretely present
- Do NOT re-report issues listed in KNOWN ISSUES"""

DEPENDENCY_SYSTEM = AGENT_SYSTEM + "\n\n" + _DEPENDENCY_DOMAIN_RULES


class DependencyAgent(BaseAgent):
    name = "dependency_agent"
    domain = "Dependency audit — CVEs, unpinned versions, outdated packages across Maven/pip/npm/Docker"
    system_prompt = DEPENDENCY_SYSTEM
    files_to_review = [
        # Maven — Spring Boot dependencies
        "Devops/springboot-backend/pom.xml",
        # pip — Python ML service
        "Devops/ml-service/requirements.txt",
        # npm — React frontend
        "Devops/react-frontend/package.json",
        # Docker base image versions
        "Devops/docker-compose.yml",
        # Individual Dockerfiles (base image pinning)
        "Devops/springboot-backend/Dockerfile",
        "Devops/ml-service/Dockerfile",
        "Devops/react-frontend/Dockerfile",
    ]


def run_dependency_agent(state: CortexState) -> dict:
    """LangGraph node: Dependency audit agent."""
    return DependencyAgent().run_node(state)
