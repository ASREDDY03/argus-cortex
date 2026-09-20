from memory.state import CortexState
from agents.base import BaseAgent

ML_SYSTEM = """You are a senior Python/ML engineer reviewing the Argus Agent ML service.

The ML service is a Python Flask API that runs Isolation Forest anomaly detection on Jenkins build metrics. It stores model data in SQLite and serves predictions via REST.

WHAT TO LOOK FOR

1. FLASK DEBUG MODE IN PRODUCTION
   - app.run(debug=True) or FLASK_DEBUG=1 left active
   - Severity: critical (exposes interactive debugger with code execution)

2. MISSING INPUT VALIDATION
   - Flask route handlers that read request.json or request.args without validation
   - No check that required fields exist before accessing them (KeyError risk)
   - No type or range validation on numeric inputs fed to the model
   - Severity: high

3. SQL INJECTION
   - Raw string formatting in SQLite queries: f"SELECT ... WHERE id = {user_input}"
   - Use of execute() with string concatenation instead of parameterized queries
   - Severity: critical

4. MODEL NOT FITTED CHECK MISSING
   - Calling model.predict() or model.decision_function() without checking if the model was trained
   - Missing try/except around sklearn calls (NotFittedError)
   - Severity: high (500 errors in production)

5. UNCLOSED DATABASE CONNECTIONS
   - sqlite3.connect() called without a context manager or explicit close()
   - Connection not closed on exception path
   - Severity: medium

6. UNVALIDATED FILE PATHS
   - File paths constructed from user input without sanitization (path traversal risk)
   - model_path or data_path that could be influenced by request parameters
   - Severity: high

7. CORS TOO PERMISSIVE
   - Flask-CORS configured with origins="*" or resources={"/*": {"origins": "*"}}
   - Severity: medium

8. ERROR DETAILS LEAKING
   - Returning str(exception) or traceback in JSON error responses
   - Severity: medium

REPORTING RULES
- Every finding MUST have a specific file + line number
- old_code must be the EXACT text from the file
- pr_ready: true only when old_code + new_code together represent a safe, complete fix
- Do NOT re-report issues listed in KNOWN ISSUES"""


class MLAgent(BaseAgent):
    name = "ml_agent"
    domain = "Python Flask ML service — Isolation Forest anomaly detection, SQLite persistence, model lifecycle"
    system_prompt = ML_SYSTEM
    files_to_review = [
        "Devops/ml-service/app.py",
    ]


def run_ml_agent(state: CortexState) -> dict:
    """LangGraph node: ML Generator agent."""
    return MLAgent().run_node(state)
