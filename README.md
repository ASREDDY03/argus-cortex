# Argus Cortex

AI agent network that continuously audits and improves [Argus Agent](https://github.com/ASREDDY03/argus-agent) — opening real GitHub PRs with code patches, tests, and risk scores.

## Architecture

```
Goal (plain English)
      │
      ▼
  PLANNER  (claude-sonnet-4-6)
  Decomposes the goal, selects agents, assigns per-agent focus
      │
      ▼  parallel fan-out
  ┌──────────────────────────────────────────────────────┐
  │  GENERATOR AGENTS  (claude-haiku-4-5, 8 in parallel) │
  │                                                      │
  │  springboot_agent    — Java/Spring Boot backend      │
  │  ml_agent            — Python ML service (Flask)     │
  │  react_agent         — React frontend                │
  │  infra_agent         — Docker Compose, K8s, Nginx    │
  │  observability_agent — Prometheus, Grafana, Alerts   │
  │  jenkins_agent       — Jenkinsfile rewrite + audit   │
  │  security_agent      — OWASP Top 10 cross-cutter     │
  │  dependency_agent    — CVE + outdated deps scanner   │
  └──────────────────────────────────────────────────────┘
      │
      ▼  fan-in
  SYNTHESIZER  (claude-sonnet-4-6)
  Cross-cutting pattern detection, coverage gap analysis,
  flags agents for a targeted second pass
      │
      ├─── coverage gaps? ──► RETRY DISPATCHER
      │                           │ (second pass, capped at 1)
      │                           └──► agents ──► SYNTHESIZER
      │
      ▼
  EVALUATOR  (claude-sonnet-4-6)
  Independently scores every finding — rejects vague ones,
  enforces confidence ≥ 70, requires concrete file + line
      │
      │  no approved findings → END
      │
      ▼  ◄── INTERRUPT (human reviews + approves here)
  HUMAN REVIEW
      │
      ▼
  TEST WRITER  (claude-haiku-4-5)
  Generates JUnit 5 / pytest / Jest tests for every patched file
      │
      ▼
  CRITIC  (claude-sonnet-4-6)
  Risk-scores each finding: low / medium / high
  Score surfaces in the PR body for reviewers
      │
      ▼
  PR CREATOR
  Commits patches + generated tests to one branch,
  opens a consolidated draft PR on Argus Agent
      │
      ▼
   END
```

## Features

- **8 parallel generator agents** — each owns a domain; all run simultaneously
- **Synthesizer** — spots cross-cutting patterns before the evaluator sees them; triggers targeted retries for coverage gaps
- **Evaluator** — independent quality gate: rejects hedged language, low-confidence findings, and unfixable suggestions
- **Human-in-the-loop** — LangGraph interrupt before any PR is opened; nothing hits GitHub without approval
- **Test Writer** — auto-generates tests (JUnit 5 / pytest / Jest) for every patched file, committed to the same PR branch
- **Critic** — risk-scores patches so reviewers know what to scrutinize vs rubber-stamp
- **Deduplication** — suppresses findings already active from a previous run
- **SHA-based skip** — files whose content hasn't changed since last review are skipped
- **Durable execution** — SQLite checkpointing via LangGraph; survives crashes and is resumable
- **LangSmith tracing** — every agent call, tool use, and state transition is observable
- **Webhook server** — ngrok tunnel receives Argus Agent push events and auto-triggers a run
- **Cron scheduling** — APScheduler runs the full audit on any cron expression
- **Slack notifications** — run start/complete events posted to a configured webhook
- **Email digest** — weekly summary of findings, approval rates, and trend data
- **Web dashboard** — live run status and finding browser at `localhost:8001`
- **Outcome feedback** — PR accept/reject rates feed back into future evaluator context

## Setup

```bash
git clone https://github.com/ASREDDY03/argus-cortex.git
cd argus-cortex
pip install -e .
cp .env.example .env   # fill in API keys
```

## Usage

```bash
# Full audit
python main.py run "Full audit of Argus Agent"

# Targeted audits
python main.py run "Review only the ML service"
python main.py run "Check infrastructure security"
python main.py run "Audit all dependencies for CVEs"

# Flags
python main.py run "goal" --dry-run    # preview findings, skip PRs
python main.py run "goal" --auto       # auto-approve all (CI/webhook mode)
python main.py run "goal" --web        # stream results to web dashboard

# Utilities
python main.py sync-prs               # sync open PR states from GitHub
python main.py outcome-report         # show accept vs reject rates per agent
python main.py history                # list recent runs
python main.py history <run-id>       # full findings for one run
python main.py watch                  # live dashboard (terminal)
python main.py trends                 # week-over-week trend report
python main.py send-digest            # send weekly email digest

# Webhook server (auto-trigger on Argus Agent push)
python webhook_server.py
```

## Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ORCHESTRATOR_MODEL` | Planner / Synthesizer / Evaluator / Critic model (default: `claude-sonnet-4-6`) |
| `AGENT_MODEL` | Generator agent + Test Writer model (default: `claude-haiku-4-5-20251001`) |
| `LANGCHAIN_API_KEY` | LangSmith API key (observability) |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing (default: `true`) |
| `GITHUB_TOKEN` | GitHub PAT for opening PRs on Argus Agent |
| `ARGUS_REPO_PATH` | Local path to the Argus Agent repo |
| `WEBHOOK_SECRET` | HMAC secret for GitHub webhook signature verification |
| `WEBHOOK_PORT` | Webhook server port (default: `8000`) |
| `DASHBOARD_PORT` | Web dashboard port (default: `8001`) |
| `SCHEDULE_CRON` | APScheduler cron expression for automatic runs (e.g. `0 9 * * 1`) |
| `SCHEDULE_GOAL` | Goal used for scheduled runs |
| `SLACK_WEBHOOK_URL` | Slack incoming webhook URL for run notifications |
| `CI_VALIDATION` | Wait for CI to pass before opening PR (default: `true`) |
| `CI_TIMEOUT` | Seconds to wait for CI (default: `600`) |

## Models

| Role | Model | Why |
|---|---|---|
| Planner, Synthesizer, Evaluator, Critic | `claude-sonnet-4-6` | Complex reasoning, cross-agent analysis, risk scoring |
| 8 Generator agents, Test Writer | `claude-haiku-4-5` | Fast, cheap, domain-specific analysis in parallel |

## Webhook Auto-Trigger

```bash
# 1. Start an ngrok tunnel
ngrok http 8000

# 2. Register the webhook on the Argus Agent repo
#    Payload URL:  https://<ngrok-id>.ngrok.io/webhook
#    Content type: application/json
#    Secret:       <matches WEBHOOK_SECRET in .env>
#    Events:       push

# 3. Start the server
python webhook_server.py
```

Every push to Argus Agent's default branch fires a targeted Cortex run scoped to the changed files.

## Scheduled Runs

Set `SCHEDULE_CRON` and `SCHEDULE_GOAL` in `.env`:

```env
SCHEDULE_CRON=0 9 * * 1       # every Monday at 09:00 UTC
SCHEDULE_GOAL=Scheduled full audit — security, CI/CD, code quality, and dependency health
```

Then start the webhook server — the scheduler runs inside the same process.
