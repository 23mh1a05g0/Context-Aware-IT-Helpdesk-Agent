# Verification Checklist & Runbook

This document provides a step-by-step verification guide with runnable commands and expected outputs to validate every capability of the Context-Aware IT Helpdesk Agent.

---

## 1. Prerequisites Check

Verify your Python environment and installed packages:

```bash
python --version
pip show fastapi tiktoken openai httpx pytest
```

Expected output:
- Python version 3.10+ (3.12 recommended)
- All packages installed and verified.

---

## 2. Automated Test Suite Execution

Run the complete test suite (103 unit and integration tests):

```bash
python -m pytest tests/ -v
```

Expected output:
```text
============================= 103 passed in 1.91s =============================
```

---

## 3. Automated Five-Turn Evaluation

Run the multi-turn evaluation script verifying pinned memory survival, topic switching, and token eviction:

```bash
python eval_5_turns.py
```

Expected output:
```text
==================================================
  5-TURN CONTEXT EVALUATION
==================================================

  [PASS] Health endpoint returns 200
  [PASS] Turn 1: HTTP 200
  [PASS] Turn 1: active_topic == general
  [PASS] Turn 2: HTTP 200
  [PASS] Turn 2: active_topic == ticket_inquiry
  [PASS] Turn 2: pinned ticket == IT-8812
  [PASS] Turn 3: HTTP 200
  [PASS] Turn 3: active_topic == wifi_issue
  [PASS] Turn 3: pinned ticket == IT-8812
  [PASS] Turn 4: HTTP 200
  [PASS] Turn 4: active_topic == wifi_issue
  [PASS] Turn 4: pinned ticket == IT-8812
  [PASS] Turn 5: HTTP 200
  [PASS] Turn 5: pinned ticket == IT-8812
  [PASS] Turn 5: response contains IT-8812
  [PASS] Token eviction occurred (recent_tokens <= budget)
  [PASS] Pinned ticket survived eviction (IT-8812)
  [PASS] Context persisted to disk
  [PASS] Persisted context has correct ticket
  [PASS] Original ticket message evicted from recent (proves eviction)

==================================================
  RESULT: PASS (20/20 checks)
==================================================
```

Verify that the process exits with status code `0`:
```bash
# Windows PowerShell
$LASTEXITCODE

# Bash / Linux / macOS
echo $?
```
Expected output: `0`

---

## 4. Manual API Verification (Local Server)

### Step 4.1: Start the API Server
In a terminal window:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Step 4.2: Health Check
```bash
curl http://localhost:8000/health
```
Expected response:
```json
{"status":"ok"}
```

### Step 4.3: Submit Ticket ID
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "manual_demo", "message": "My ticket is IT-9921"}'
```
Expected response:
```json
{
  "response": "I've noted your ticket ID IT-9921. I don't have access to the university's ticketing system to look up its details, but I'll keep this ticket ID associated with our conversation. How can I help you today?",
  "active_topic": "ticket_inquiry"
}
```

### Step 4.4: Inspect Context State
```bash
curl http://localhost:8000/v1/context/manual_demo
```
Expected response:
```json
{
  "session_id": "manual_demo",
  "active_topic": "ticket_inquiry",
  "pinned": {
    "ticket_id": "IT-9921"
  },
  "recent": [
    {
      "role": "user",
      "content": "My ticket is IT-9921",
      "tokens": 8
    },
    {
      "role": "assistant",
      "content": "I've noted your ticket ID IT-9921...",
      "tokens": 44
    }
  ],
  "total_recent_tokens": 52
}
```

### Step 4.5: Topic Shift to WiFi
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "manual_demo", "message": "My WiFi in the library dropped. I am using a Mac."}'
```
Expected response:
```json
{
  "response": "Let's troubleshoot your WiFi connection on Mac. Here are the steps:\n\n1. Click the WiFi icon in the menu bar and toggle WiFi off, then back on...\n",
  "active_topic": "wifi_issue"
}
```

### Step 4.6: Ask for Ticket Number (Recall Test)
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "manual_demo", "message": "Can you confirm my original ticket number?"}'
```
Expected response:
```json
{
  "response": "Your ticket number is IT-9921. I still have it saved from our conversation. However, I don't have access to the university's ticketing system to retrieve its live status.",
  "active_topic": "ticket_inquiry"
}
```

### Step 4.7: Verify Session Isolation
Submit a different ticket in a second session:
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session_B", "message": "My ticket is IT-4444"}'
```

Verify that session B has IT-4444:
```bash
curl http://localhost:8000/v1/context/session_B
```
Expected response: `"ticket_id": "IT-4444"`

Verify that original session `manual_demo` still has IT-9921 (no crosstalk):
```bash
curl http://localhost:8000/v1/context/manual_demo
```
Expected response: `"ticket_id": "IT-9921"`

### Step 4.8: Verify Persistence on Disk
Inspect the persisted JSON file on disk:
```bash
# Windows PowerShell
Get-Content logs/context_manual_demo.json

# Linux / macOS
cat logs/context_manual_demo.json
```
Expected: Valid formatted JSON containing `pinned.ticket_id == "IT-9921"`.

### Step 4.9: Verify 404 for Non-Existent Session
```bash
curl http://localhost:8000/v1/context/nonexistent_xyz
```
Expected response (HTTP 404):
```json
{"detail":"Session 'nonexistent_xyz' not found."}
```

### Step 4.10: Verify 422 for Invalid Requests
Missing message:
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test"}'
```
Expected status: `HTTP 422 Unprocessable Entity`

---

## 5. Docker Verification

### Build & Run via Docker Compose
```bash
# Start service
docker compose up --build -d

# Verify container status
docker compose ps

# Check health endpoint
curl http://localhost:8000/health

# Run test query
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "docker_test", "message": "My ticket is IT-7777"}'

# Verify context file created in mounted host directory
ls -l logs/context_docker_test.json

# Stop service
docker compose down
```

---

## 6. Definition of Done Checklist

| Requirement | Status |
|---|---|
| `POST /v1/chat` works | **PASSED** |
| `GET /v1/context/{session_id}` works | **PASSED** |
| `GET /health` works | **PASSED** |
| Session state is maintained | **PASSED** |
| Session isolation works | **PASSED** |
| Ticket IDs are extracted (`IT-XXXX`) | **PASSED** |
| Ticket IDs are pinned | **PASSED** |
| Active topic is tracked across dialogue turns | **PASSED** |
| Recent messages are token-counted via `tiktoken` | **PASSED** |
| FIFO token eviction works | **PASSED** |
| Pinned memory survives token eviction | **PASSED** |
| Prompt builder injects pinned facts and active topic | **PASSED** |
| Context is persisted to human-readable JSON | **PASSED** |
| LLM integration exists (OpenAI) | **PASSED** |
| Mock LLM mode exists (deterministic for tests/demo) | **PASSED** |
| LLM failure causes rollback (pre-request snapshot) | **PASSED** |
| Five-turn evaluation exists (`eval_5_turns.py`) | **PASSED** |
| Five-turn evaluation passes (20/20 checks, exit 0) | **PASSED** |
| Comprehensive unit and integration tests exist (103 tests) | **PASSED** |
| All tests pass | **PASSED** |
| `Dockerfile` exists | **PASSED** |
| `docker-compose.yml` exists | **PASSED** |
| `.dockerignore` exists | **PASSED** |
| `.env.example` exists | **PASSED** |
| No secrets committed | **PASSED** |
| `README.md` complete with Mermaid diagram | **PASSED** |
| `VERIFY.md` complete with runnable commands | **PASSED** |
| Modular, clean, production-grade Python architecture | **PASSED** |
