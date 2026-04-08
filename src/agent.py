"""
ResearchAgent — CORE-Bench purple agent.

Real CORE-Bench protocol (discovered from ab-shetty/agentbeats-corebench):

  The green agent OWNS the code/data environment.
  It exposes tools (execute_bash, inspect_file_as_text, etc.) by parsing
  <json>...</json> tags from the purple agent's TEXT responses.

  Conversation flow (multi-turn A2A, same context_id):
    Turn 1:  Green → "Task description + available tools + instructions"
             Purple → "<json>{"name": "execute_bash", "arguments": {"command": "ls"}}</json>"
    Turn 2:  Green → "Exit Code: 0\ncode/ data/ results/"
             Purple → "<json>{"name": "inspect_file_as_text", ...}</json>"
    Turn N:  Green → "file contents..."
             Purple → "<json>{"name": "FINAL_ANSWER", "arguments": {"content": {"Q?": "A"}}}</json>"

  So the purple agent:
  1. Maintains full conversation history across A2A turns (via context_id)
  2. Each run() call = one LLM decision turn
  3. Outputs a TEXT response with a <json>...</json> block (one tool call at a time)
  4. When FINAL_ANSWER is detected → emit DataPart artifact + complete
"""

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_text_message

load_dotenv()

# ---------------------------------------------------------------------------
# System prompt — teaches the model the <json> tool-call format
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a research reproduction agent for CORE-Bench (Computational Reproducibility Benchmark).

## Your situation
- You are inside a scientific paper's code repository (CodeOcean capsule).
- You have access to tools provided by the evaluation system.
- Tools are executed by the evaluator — you call them by outputting JSON.

## How to call tools
Output EXACTLY ONE tool call per message, wrapped in <json>...</json> tags:

<json>
{"name": "execute_bash", "arguments": {"command": "ls -la"}}
</json>

Available tools (the evaluator will list them in the task):
- execute_bash: Run shell commands. Use `cd dir && command` to chain (state is NOT preserved).
- inspect_file_as_text: Read file contents as text.
- query_vision_language_model: Analyze plots/images from the paper.
- web_search: Search for documentation if needed.

## How to submit final answer
When you have all answers, output:

<json>
{"name": "FINAL_ANSWER", "arguments": {"content": {"Exact question text?": "exact_value", ...}}}
</json>

## Strategy for CORE-Bench tasks

### Easy (read results dir):
1. `execute_bash` → `ls environment/results/` or `find . -name "*.csv" -o -name "*.json"`
2. Read each result file with `inspect_file_as_text`
3. Extract the specific values asked for
4. Submit FINAL_ANSWER

### Medium (reproduce with REPRODUCING.md):
1. Read REPRODUCING.md: `inspect_file_as_text` → path="REPRODUCING.md"
2. Follow the exact steps (install deps, run commands)
3. Chain commands: `cd code && pip install -r requirements.txt && python run.py`
4. Capture output and extract metrics
5. Submit FINAL_ANSWER

### Hard (no instructions — figure it out):
1. List all files: `execute_bash` → `find . -maxdepth 3 -type f | head -50`
2. Read README, setup.py, requirements.txt
3. Find main entry point (main.py, run.py, Makefile)
4. Install deps and run
5. Parse outputs for the requested metrics
6. Submit FINAL_ANSWER

## Critical rules
- Answer keys MUST match the question text EXACTLY (copy-paste)
- Answer values MUST be real extracted values — no guesses, no summaries
- ONE tool call per response
- If a command fails, read the error and try a fix
- Always submit FINAL_ANSWER even if some values are "unknown"
"""

# Regex to extract <json>...</json> block from model output
JSON_TAG_RE = re.compile(r"<json>\s*(.*?)\s*</json>", re.DOTALL)


class ResearchAgent:
    def __init__(self):
        self.model = os.getenv("AGENT_MODEL", "gpt-4o-mini")
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Full conversation history for this context (multi-turn)
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.turn = 0

    # ------------------------------------------------------------------
    # Called on every A2A message in this context
    # ------------------------------------------------------------------

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        self.turn += 1

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"[turn {self.turn}] Processing..."),
        )

        # Append the incoming message (task description or tool result from green agent)
        self.messages.append({"role": "user", "content": input_text})

        # One LLM call per turn
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.0,
            max_tokens=2048,
        )

        assistant_text = response.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": assistant_text})

        # Parse the <json>...</json> block
        action = _parse_json_action(assistant_text)

        if action and action.get("name") == "FINAL_ANSWER":
            # Emit answer as DataPart artifact
            content = action.get("arguments", {}).get("content", {})
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"[turn {self.turn}] Submitting FINAL_ANSWER with {len(content)} keys"),
            )
            await updater.add_artifact(
                parts=[Part(root=DataPart(data={
                    "final_answer": content,
                    "turns": self.turn,
                }))],
                name="FinalAnswer",
            )
            await updater.complete()

        else:
            # Regular tool call — return text so green agent can parse <json> and execute
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"[turn {self.turn}] Tool call: {action.get('name', '?') if action else 'text response'}"
                ),
            )
            await updater.add_artifact(
                parts=[Part(root=TextPart(kind="text", text=assistant_text))],
                name=f"ToolCall_t{self.turn}",
            )
            # Don't call updater.complete() — green agent will send the next message


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_json_action(text: str) -> dict[str, Any] | None:
    """Extract and parse the first <json>...</json> block."""
    match = JSON_TAG_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
