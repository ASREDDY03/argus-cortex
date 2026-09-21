from memory.state import CortexState
from agents.base import BaseAgent, AGENT_SYSTEM

_REACT_DOMAIN_RULES = """
You are a senior React engineer reviewing the Argus Agent React 18 frontend.

The frontend is a Jenkins dashboard -- it polls build status, renders job lists, shows a Grafana tab, and has a job detail drawer. It calls a Spring Boot REST API using fetch/axios.

WHAT TO LOOK FOR

1. STALE CLOSURE BUGS IN useEffect
   - useEffect that reads state or props but is missing those values in the dependency array
   - Interval or timeout set in useEffect without cleanup (return () => clearInterval(...))
   - Severity: high (stale data bugs, memory leaks)

2. MISSING ERROR HANDLING ON FETCH/ASYNC
   - fetch() calls without .catch() or try/catch
   - async/await in useEffect without try/catch
   - No error state shown to the user when an API call fails
   - Severity: high

3. XSS VECTORS
   - dangerouslySetInnerHTML used with any variable content (not a static string literal)
   - Severity: critical

4. CREDENTIALS IN LOCALSTORAGE
   - JWT tokens, API keys, or session tokens stored in localStorage or sessionStorage
   - Should use httpOnly cookies instead
   - Severity: high

5. MISSING LOADING STATES
   - API calls that update state but show no loading indicator (user sees stale or blank data)
   - Severity: low

6. RACE CONDITIONS
   - useEffect that kicks off an async call but does not cancel it on unmount
   - Multiple concurrent requests that could resolve out of order
   - Severity: medium

7. PROP DRILLING / MISSING MEMOIZATION
   - Expensive computations (sorting, filtering large arrays) not wrapped in useMemo
   - Callback functions recreated on every render and passed as props without useCallback
   - Severity: low

8. ACCESSIBILITY GAPS
   - Interactive elements (<div onClick=...>) missing role= and keyboard handler
   - Images missing alt= attribute
   - Form inputs missing associated <label>
   - Severity: low

REPORTING RULES
- Every finding MUST have a specific file + line number
- old_code must be the EXACT text from the file
- pr_ready: true only when old_code + new_code represent a safe, complete fix
- Do NOT re-report issues listed in KNOWN ISSUES"""

REACT_SYSTEM = AGENT_SYSTEM + "\n\n" + _REACT_DOMAIN_RULES


class ReactAgent(BaseAgent):
    name = "react_agent"
    domain = "React 18 frontend — Jenkins dashboard, Grafana monitoring tab, job detail drawer"
    system_prompt = REACT_SYSTEM
    files_to_review = [
        "Devops/react-frontend/src/components/JenkinsDashboardComponent.jsx",
        "Devops/react-frontend/src/services/JenkinsService.js",
    ]


def run_react_agent(state: CortexState) -> dict:
    """LangGraph node: React Generator agent."""
    return ReactAgent().run_node(state)
