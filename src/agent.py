"""
ResearchAgent — CORE-Bench purple agent.

Workflow per task:
  1. Parse incoming message (may contain paper text + repo path/zip URL).
  2. Set up a per-task workspace (temp dir that persists across turns in same context).
  3. Run an OpenAI tool-calling loop:
       - Explore the workspace (list files, read README/requirements)
       - Understand what results need to be reproduced
       - Install dependencies & run code
       - Capture outputs and compare with expected results
       - Call report_results() to finalize
  4. Return results as an artifact.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message

from tools import TOOLS, dispatch_tool

load_dotenv()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a research reproduction agent competing in CORE-Bench (Computational Reproducibility Benchmark).

Your goal: given a scientific paper's code repository, reproduce its key computational results.

## Workflow

### Step 1 — Get the code
If the task provides a GitHub URL or zip link, call clone_repo() first.
Otherwise the repo is already in the workspace.

### Step 2 — Explore
- list_files('.', recursive=True) to see the full structure
- Read these files (if they exist): README.md, REPRODUCE.md, run.sh, Makefile,
  requirements.txt, environment.yml, setup.py, pyproject.toml
- Identify the language (Python/R/Julia/etc.) and entry points

### Step 3 — Understand what to reproduce
- Read the task description carefully — it specifies WHICH metrics/tables to reproduce
- Note expected values (numbers, percentages, p-values, accuracy scores, etc.)

### Step 4 — Setup environment
- Python: `pip install -r requirements.txt` (timeout 180s)
- R: `Rscript -e "install.packages(...)"`
- Conda env.yml: `conda env create -f environment.yml && conda run -n <env> ...`
- If setup fails, try installing packages one by one or find minimal subset

### Step 5 — Execute
- Run main script(s) per README instructions
- Capture ALL output: stdout, stderr, result files (.csv, .json, .txt, .log)
- If a run fails: read the error, fix it, retry
- For long-running jobs: use timeout_seconds=600 or more
- Look for pre-computed results if full training is infeasible

### Step 6 — Extract metrics
- search_in_files() for metric names, table numbers, or specific values
- Read result files directly
- Parse stdout for printed metrics

### Step 7 — Report
- Call report_results() with reproduced_values and expected_values
- Be precise: include exact numbers, not just "similar"
- Even partial results are valuable — report what you got

## Key rules
- Work ONLY inside the workspace directory (no absolute paths outside it)
- Always call report_results() at the end, even on failure
- If a command takes >120s, re-issue with higher timeout_seconds
- Don't guess results — only report what you actually measured
- If code requires GPU/large data and fails, document this in errors[]
"""

# Max tool-calling iterations before giving up
MAX_ITERATIONS = 30


class ResearchAgent:
    def __init__(self):
        self.model = os.getenv("AGENT_MODEL", "gpt-4o-mini")
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Workspace persists for the lifetime of this agent instance (one context)
        self._workspace: Path | None = None
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._final_result: dict | None = None

    @property
    def workspace(self) -> Path:
        if self._workspace is None:
            self._workspace = Path(tempfile.mkdtemp(prefix="agentx_research_"))
        return self._workspace

    def cleanup(self):
        if self._workspace and self._workspace.exists():
            shutil.rmtree(self._workspace, ignore_errors=True)
            self._workspace = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting research reproduction... workspace={self.workspace}"),
        )

        # Append user turn
        self.messages.append({"role": "user", "content": input_text})

        # Agentic loop
        result = await self._run_loop(updater)

        # Return artifact
        from a2a.types import DataPart, Part
        await updater.add_artifact(
            parts=[Part(root=DataPart(data=result))],
            name="ReproductionResult",
        )

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    async def _run_loop(self, updater: TaskUpdater) -> dict:
        for iteration in range(MAX_ITERATIONS):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"[step {iteration + 1}] Calling model..."),
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=4096,
            )

            choice = response.choices[0]
            assistant_msg = choice.message

            # Append assistant message to history
            self.messages.append(assistant_msg.model_dump(exclude_none=True))

            # ── Finished: no more tool calls ──────────────────────────
            if choice.finish_reason == "stop":
                # Model gave a text answer without calling report_results
                return {
                    "success": False,
                    "match_summary": assistant_msg.content or "(no response)",
                    "reproduced_values": {},
                    "errors": ["Agent stopped without calling report_results"],
                }

            # ── Tool calls ────────────────────────────────────────────
            if not assistant_msg.tool_calls:
                break

            tool_results = []
            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"[tool] {tool_name}({list(tool_args.keys())})"),
                )

                # report_results is terminal
                if tool_name == "report_results":
                    self._final_result = tool_args
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Results recorded. Task complete.",
                    })
                    self.messages.extend(tool_results)
                    return tool_args  # ← exit loop early

                # All other tools
                result_str = dispatch_tool(self.workspace, tool_name, tool_args)

                # Truncate very long results before adding to history
                if len(result_str) > 6000:
                    result_str = result_str[:6000] + "\n... (output truncated)"

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

            self.messages.extend(tool_results)

        # Exhausted iterations
        return {
            "success": False,
            "match_summary": "Agent exhausted max iterations without completing reproduction",
            "reproduced_values": {},
            "errors": [f"Hit MAX_ITERATIONS={MAX_ITERATIONS}"],
        }
