"""
Security Agent — claude-haiku-4-5
Cross-cutting OWASP vulnerability scanner.

Spans all layers of the stack:
  - Spring Boot controllers/services — injection, broken auth, exposed actuators
  - Python ML service (Flask) — command injection, debug mode, missing validation
  - React frontend — XSS, insecure token storage
  - nginx — missing security headers, TLS misconfiguration
  - Docker Compose — exposed ports, hardcoded env vars
  - Jenkinsfile — credentials mishandling in CI

Context files cover the full attack surface rather than one domain.
"""
from memory.state import CortexState
from agents.base import BaseAgent

SECURITY_SYSTEM = """You are a senior application security engineer performing an OWASP Top 10 review of the Argus Agent codebase.

Argus Agent is a Spring Boot backend + React frontend + Python ML (Flask) service, deployed via Docker Compose on Kubernetes, proxied by nginx, with a Jenkins CI pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO LOOK FOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. HARDCODED SECRETS
   - Credentials, API keys, tokens, passwords written directly in source (not read from env)
   - Look for: password =, secret =, apiKey =, token =, Authorization: Basic <literal>
   - Severity: critical

2. INJECTION
   - SQL injection: string concatenation inside queries instead of parameterized statements
   - Command injection: Runtime.exec() / ProcessBuilder / subprocess.run(shell=True) with user-controlled input
   - SSTI / eval() on user input in Python
   - Severity: critical

3. BROKEN AUTHENTICATION & AUTHORIZATION
   - Spring REST endpoints missing @PreAuthorize or SecurityConfig rules
   - No JWT validation — token accepted without signature check
   - Missing authentication on admin or sensitive endpoints
   - Severity: high

4. SENSITIVE DATA EXPOSURE
   - Credentials or tokens written to logs (log.info with password variable, logger.debug with headers)
   - Sensitive fields (password, secret, token) returned in API response bodies / JSON serialization
   - Full stack traces exposed in error responses reaching the client
   - Severity: high

5. SECURITY MISCONFIGURATION
   - Spring Boot Actuator endpoints (/actuator/env, /actuator/heapdump) exposed without auth
   - CORS configured with wildcard origin ("*") on non-public endpoints
   - Flask debug=True or app.run(debug=True) left in production code
   - Severity: high

6. XSS (Cross-Site Scripting)
   - React dangerouslySetInnerHTML used with unsanitized user input
   - innerHTML assignment in plain JS
   - Severity: high

7. INSECURE TRANSPORT & MISSING SECURITY HEADERS
   - nginx missing: Strict-Transport-Security, X-Frame-Options, X-Content-Type-Options,
     Content-Security-Policy, Referrer-Policy
   - Internal service-to-service calls over plain HTTP where HTTPS is available
   - Severity: medium

8. EXPOSED PORTS & OVER-PRIVILEGED CONTAINERS
   - Docker Compose services binding internal ports (DB, cache, internal APIs) to 0.0.0.0 unnecessarily
   - Services running as root with no user: directive
   - Severity: medium

9. INSECURE DIRECT OBJECT REFERENCE (IDOR)
   - REST endpoints that accept a resource ID (e.g. /builds/{id}) without verifying the caller owns it
   - Severity: medium

10. CREDENTIALS IN CI/CD
    - Jenkinsfile using hardcoded Docker Hub username or registry URL instead of credentials()
    - Secrets echoed in shell steps
    - Severity: high

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPORTING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Every finding MUST have a specific file + line number
- old_code must be the EXACT text from the file (copy-paste it) — it is used for find-and-replace
- old_code must be unique enough in the file to avoid false replacements
- pr_ready: true only when old_code + new_code together represent a safe, complete fix
- Do NOT fabricate vulnerabilities — only report what is concretely present in the code
- Do NOT re-report issues listed in KNOWN ISSUES"""


class SecurityAgent(BaseAgent):
    name = "security_agent"
    domain = "Security — OWASP Top 10, hardcoded secrets, injection, broken auth, missing headers, exposed ports"
    system_prompt = SECURITY_SYSTEM
    files_to_review = [
        # Spring Boot — primary attack surface
        "Devops/springboot-backend/src/main/java/com/example/crudspring/controllers/JenkinsController.java",
        "Devops/springboot-backend/src/main/java/com/example/crudspring/services/JenkinsService.java",
        # Python ML service — Flask endpoints, subprocess usage
        "Devops/ml-service/app.py",
        # Infrastructure — security headers, exposed ports
        "Devops/nginx/nginx.conf",
        "Devops/docker-compose.yml",
        # CI pipeline — credentials handling
        "Devops/Jenkinsfile",
    ]


def run_security_agent(state: CortexState) -> dict:
    """LangGraph node: Security scanner agent."""
    if "security_agent" not in state.get("agents_to_run", []):
        return {"findings": []}
    agent = SecurityAgent()
    discovered = state.get("agent_files", {}).get("security_agent", [])
    if discovered:
        agent.files_to_review = discovered
    focus = state.get("agent_focus", {}).get("security_agent", "")
    findings = agent.analyze(state["goal"], focus=focus)
    return {"findings": findings}
