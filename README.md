# Argus Cortex

AI agent network that audits and improves [Argus Agent](https://github.com/ASREDDY03/argus-agent) using the **Planner → Generator → Evaluator** pattern.

## Architecture

```
Goal (plain English)
      ↓
  PLANNER (claude-sonnet-4-6)
  Decomposes goal, selects agents
      ↓
  ┌─────────────────────────────────────┐
  │  GENERATOR AGENTS (claude-haiku-4-5) │  ← run in parallel
  │  springboot_agent                   │
  │  ml_agent                           │
  │  react_agent                        │
  │  infra_agent                        │
  │  observability_agent                │
  └─────────────────────────────────────┘
      ↓
  EVALUATOR (claude-sonnet-4-6)
  Scores every finding, rejects vague ones
      ↓
  Approved findings → PRs on Argus Agent
```

## Setup

```bash
# 1. Clone
git clone https://github.com/ASREDDY03/argus-cortex.git
cd argus-cortex

# 2. Install dependencies
pip install -e .

# 3. Configure
cp .env.example .env
# Fill in your API keys in .env

# 4. Run
python main.py "Full audit of Argus Agent"
python main.py "Review only the ML service"
python main.py "Check infrastructure security"
```

## Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key |
| `ORCHESTRATOR_MODEL` | Planner + Evaluator model (default: claude-sonnet-4-6) |
| `AGENT_MODEL` | Generator agent model (default: claude-haiku-4-5-20251001) |
| `LANGCHAIN_API_KEY` | LangSmith API key (observability) |
| `GITHUB_TOKEN` | GitHub PAT for opening PRs |
| `ARGUS_REPO_PATH` | Local path to the Argus Agent repo |

## Models

| Role | Model | Why |
|---|---|---|
| Planner | claude-sonnet-4-6 | Complex goal decomposition |
| Generator agents | claude-haiku-4-5 | Fast, cheap, domain-specific analysis |
| Evaluator | claude-sonnet-4-6 | Independent quality assessment |

## Features

- **Durable execution** — LangGraph checkpointing, resume after crash
- **Human-in-the-loop** — findings evaluated before any PR is opened
- **Parallel agents** — all generators run simultaneously
- **LangSmith tracing** — full observability of every agent decision
