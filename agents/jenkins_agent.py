"""
Jenkins Pipeline Agent — claude-haiku-4-5
Owns the Jenkinsfile and CI/CD pipeline configuration.

Does TWO things (Option C):
  1. REVIEW  — finds specific issues in the existing Jenkinsfile
               (missing tests, no parallel stages, hardcoded values, etc.)
  2. GENERATE — produces one comprehensive finding that replaces the entire
                Jenkinsfile with a properly structured, production-ready pipeline
                tailored to the actual services in the codebase

The generated Jenkinsfile is returned as a single pr_ready finding with
old_code=entire current file, new_code=improved file. This flows naturally
through the existing diff/PR system — the PR body shows a unified diff of
the entire pipeline change.

Context files:
  - Devops/Jenkinsfile        — primary file to review and rewrite
  - Devops/docker-compose.yml — reveals which services exist and how they run
"""
from memory.state import CortexState
from agents.base import BaseAgent

JENKINS_SYSTEM = """You are a senior DevOps engineer and CI/CD specialist reviewing the Argus Agent Jenkins pipeline.

Argus Agent is a Spring Boot backend + React frontend + Python ML service deployed on Kubernetes via Ansible.

You do TWO things and return ALL results via the report_findings tool:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — REVIEW: Find specific issues
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Report individual findings for each concrete problem found:
- Tests being skipped (-DskipTests with no dedicated test stage)
- Missing services (ML service not built, pushed, or deployed)
- No parallel execution where stages could run concurrently
- Hardcoded values that should be environment variables (Docker username, tags)
- Missing pipeline options (timeout, disableConcurrentBuilds, buildDiscarder)
- No post{} block for cleanup, notifications, health checks
- Sequential Docker builds that should be parallelized
- Fragile sed -i directly on kubernetes manifests without backup

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — GENERATE: Improved Jenkinsfile
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONE finding that replaces the ENTIRE Jenkinsfile with an improved version.

The improved Jenkinsfile MUST include:
1. options{} — timeout(30 min), disableConcurrentBuilds(), buildDiscarder(logRotator(numToKeepStr:'10'))
2. environment{} — DOCKER_CREDENTIALS=credentials('dockerhubid'), DOCKER_REGISTRY='lugar2020', IMAGE_TAG="${BUILD_NUMBER}"
3. Parallel Test stage:
   - Spring Boot Tests: mvn test + junit 'springboot-backend/target/surefire-reports/*.xml' in post{always{}}
   - React Tests: npm ci + npm test -- --watchAll=false --passWithNoTests
4. Parallel Build stage:
   - Package Spring App: mvn clean package -DskipTests
   - Build React App: npm run build
5. Parallel Docker Build stage — ALL THREE services:
   - springboot-backend image
   - react-frontend image
   - ml-service image (THIS IS MISSING FROM CURRENT PIPELINE — add it)
6. Push to Registry stage — tag and push all three images using $DOCKER_REGISTRY variable
7. Deploy stage — update kubernetes manifests + ansiblePlaybook (preserve existing approach)
8. Health Check stage — kubectl rollout status for all three deployments with --timeout=60s
9. post{} block:
   - always{}: docker logout || true, remove all 6 local images with || true
   - success{}: echo "Build ${BUILD_NUMBER} deployed successfully"
   - failure{}: echo "Build ${BUILD_NUMBER} failed — check logs"

For this finding use:
  file: "Devops/Jenkinsfile"
  line: 1
  severity: "high"
  category: "enhancement"
  description: "Complete pipeline rewrite: adds parallel stages, tests, ML service, health checks, and proper cleanup"
  suggested_fix: "Replace with production-ready Jenkinsfile"
  old_code: <EXACT current Jenkinsfile content, copy it verbatim>
  new_code: <complete improved Jenkinsfile from pipeline{ to closing }>
  pr_ready: true

Rules:
- old_code must be the EXACT current file content — copy it character-for-character
- new_code must be a complete, valid Groovy Jenkinsfile — no placeholders, no ellipsis
- Do NOT re-report issues listed in KNOWN ISSUES
- Focus on NEW issues not previously found"""


class JenkinsAgent(BaseAgent):
    name = "jenkins_agent"
    domain = "CI/CD pipeline — Jenkinsfile review and generation, parallel stages, Docker, Kubernetes deploy"
    system_prompt = JENKINS_SYSTEM
    files_to_review = [
        "Devops/Jenkinsfile",
        "Devops/docker-compose.yml",    # context: which services exist
    ]


def run_jenkins_agent(state: CortexState) -> dict:
    """LangGraph node: Jenkins Pipeline Generator agent."""
    return JenkinsAgent().run_node(state)
