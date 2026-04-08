"""
Multi-turn CORE-Bench green agent simulator.

Simulates what the real CORE-Bench green agent does:
  1. Sends task description (with tools list) to purple agent
  2. Purple agent responds with <json>{"name": "tool", "arguments": {...}}</json>
  3. Simulator executes the tool locally, sends result back
  4. Repeat until FINAL_ANSWER or max turns

Usage:
    python test_agent.py [--scenario easy|hard]
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import DataPart, Message, Part, Role, Task, TextPart


BASE_URL = "http://127.0.0.1:9099"
MAX_TURNS = 20

# Regex matching the purple agent's <json>...</json> response
JSON_TAG_RE = re.compile(r"<json>\s*(.*?)\s*</json>", re.DOTALL)

# ──────────────────────────────────────────────────────────────────────────────
# Simulated workspace — green agent owns the environment
# ──────────────────────────────────────────────────────────────────────────────

def build_workspace_easy() -> Path:
    """
    Easy scenario: results already exist in environment/results/.
    Agent just needs to read them.
    """
    ws = Path(tempfile.mkdtemp(prefix="corebench_easy_"))
    results = ws / "environment" / "results"
    results.mkdir(parents=True)

    # Simulated result files from a "paper"
    (results / "metrics.json").write_text(json.dumps({
        "accuracy": 0.923,
        "f1_score": 0.911,
        "precision": 0.934,
        "recall": 0.889,
        "dataset": "CIFAR-10",
        "epochs": 50,
    }, indent=2))

    (results / "summary.txt").write_text(
        "Training completed.\n"
        "Best validation accuracy: 92.3%\n"
        "F1 score: 0.911\n"
        "Model: ResNet-18 with data augmentation\n"
    )

    return ws


def build_workspace_hard() -> Path:
    """
    Hard scenario: no instructions, agent must figure out how to run the code.
    """
    ws = Path(tempfile.mkdtemp(prefix="corebench_hard_"))
    code = ws / "code"
    code.mkdir()

    (code / "requirements.txt").write_text("numpy\nscipy\n")
    (code / "README.md").write_text(
        "# Mean Absolute Error Experiment\n\n"
        "Computes MAE for a set of predictions vs ground truth.\n\n"
        "## Run\n```\npython run_experiment.py\n```\n"
    )
    (code / "run_experiment.py").write_text(
        "import numpy as np\n"
        "np.random.seed(42)\n"
        "y_true = np.array([3.2, 5.1, 2.8, 7.4, 4.0])\n"
        "y_pred = np.array([3.0, 5.5, 2.9, 7.1, 4.2])\n"
        "mae = np.mean(np.abs(y_true - y_pred))\n"
        "rmse = np.sqrt(np.mean((y_true - y_pred)**2))\n"
        "print(f'MAE: {mae:.4f}')\n"
        "print(f'RMSE: {rmse:.4f}')\n"
        "print(f'Samples: {len(y_true)}')\n"
    )

    return ws


TASK_TEMPLATES = {
    "easy": {
        "build_workspace": build_workspace_easy,
        "prompt": """# Task: Corebench Easy

## Objective
You should read through the files in the `environment/results` directory to answer the following questions. **Do not run or execute any code.** All answers can be obtained by reading the results files.

## Questions to answer
- "What is the accuracy of the model?"
- "What is the F1 score?"
- "How many epochs was the model trained for?"

## Your Goal
Your submitted answer must be a JSON object where keys are the EXACT question strings above and values are the exact answers extracted from the files.

## Available Tools
- execute_bash: Run shell commands (e.g. ls, find, cat)
- inspect_file_as_text: Read a file. Arguments: {"path": "relative/path"}

## Response Format
Output ONE tool call per response, wrapped in <json>...</json> tags:
<json>
{"name": "execute_bash", "arguments": {"command": "ls environment/results/"}}
</json>

When ready to submit:
<json>
{"name": "FINAL_ANSWER", "arguments": {"content": {"What is the accuracy of the model?": "0.923", ...}}}
</json>

Begin by exploring the results directory.
""",
    },
    "hard": {
        "build_workspace": build_workspace_hard,
        "prompt": """# Task: Corebench Hard

## Objective
Your goal is to test the computational reproducibility of the repository in your current directory (a scientific paper's code). You need to install the requirements and run the code to answer the following questions.

## Questions to answer
- "What is the MAE (Mean Absolute Error)?"
- "What is the RMSE?"
- "How many samples were used?"

## Your Goal
Your submitted answer must be a JSON object where keys are EXACT question strings and values are the exact computed values.

## Available Tools
- execute_bash: Run shell commands. Shell state is NOT preserved between calls — chain with &&.
- inspect_file_as_text: Read a file. Arguments: {"path": "relative/path"}

## Response Format
<json>
{"name": "execute_bash", "arguments": {"command": "..."}}
</json>

Or to submit:
<json>
{"name": "FINAL_ANSWER", "arguments": {"content": {"What is the MAE (Mean Absolute Error)?": "0.2200", ...}}}
</json>

Begin by exploring the directory structure.
""",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Tool executor (runs on green agent side)
# ──────────────────────────────────────────────────────────────────────────────

def execute_tool(name: str, arguments: dict, workspace: Path) -> str:
    if name == "execute_bash":
        cmd = arguments.get("command", "")
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=str(workspace),
                capture_output=True, text=True, timeout=60,
            )
            out = result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout
            err = result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
            parts = [f"Exit Code: {result.returncode}"]
            if out:
                parts.append(out)
            if err:
                parts.append(f"stderr: {err}")
            return "\n".join(parts)
        except subprocess.TimeoutExpired:
            return "Exit Code: 1\nERROR: command timed out"
        except Exception as e:
            return f"Exit Code: 1\nERROR: {e}"

    elif name == "inspect_file_as_text":
        path = arguments.get("path", "")
        target = workspace / path
        if not target.exists():
            return f"ERROR: file not found: {path}"
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            if len(content) > 4000:
                content = content[:4000] + "\n... (truncated)"
            return content
        except Exception as e:
            return f"ERROR: {e}"

    elif name == "web_search":
        return "Web search not available in test environment."

    elif name == "query_vision_language_model":
        return "Vision model not available in test environment."

    else:
        return f"ERROR: unknown tool '{name}'"


# ──────────────────────────────────────────────────────────────────────────────
# Green agent simulator
# ──────────────────────────────────────────────────────────────────────────────

def parse_action(text: str) -> dict | None:
    match = JSON_TAG_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def extract_text_from_event(event) -> tuple[str, str | None]:
    """Returns (text_content, context_id)"""
    if isinstance(event, Message):
        parts_text = []
        for p in event.parts:
            if isinstance(p.root, TextPart):
                parts_text.append(p.root.text)
        return "\n".join(parts_text), event.context_id

    elif isinstance(event, tuple):
        task_obj, _ = event
        cid = task_obj.context_id
        texts = []
        if task_obj.status.message:
            for p in task_obj.status.message.parts:
                if isinstance(p.root, TextPart):
                    texts.append(p.root.text)
        if task_obj.artifacts:
            for artifact in task_obj.artifacts:
                for p in artifact.parts:
                    if isinstance(p.root, TextPart):
                        texts.append(p.root.text)
                    elif isinstance(p.root, DataPart):
                        texts.append(json.dumps(p.root.data))
        return "\n".join(texts), cid

    return "", None


async def run_simulation(scenario: str):
    cfg = TASK_TEMPLATES[scenario]
    workspace = cfg["build_workspace"]()
    initial_prompt = cfg["prompt"]

    print(f"[sim] Workspace: {workspace}")
    print(f"[sim] Scenario: {scenario}\n{'='*60}\n")

    async with httpx.AsyncClient(timeout=120) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=BASE_URL)
        card = await resolver.get_agent_card()
        factory = ClientFactory(ClientConfig(httpx_client=httpx_client, streaming=True))
        client = factory.create(card)

        context_id = None

        # Turn 0: send initial task
        current_message = initial_prompt

        for turn in range(MAX_TURNS):
            print(f"\n--- TURN {turn + 1} {'(initial)' if turn == 0 else ''} ---")
            print(f"[-> purple] {current_message[:200].strip()}{'...' if len(current_message) > 200 else ''}")

            # Send message to purple agent
            msg = Message(
                kind="message",
                role=Role.user,
                parts=[Part(root=TextPart(kind="text", text=current_message))],
                message_id=uuid4().hex,
                context_id=context_id,
            )

            # Collect response
            last_text = ""
            last_data = None
            final_answer = None

            async for event in client.send_message(msg):
                if isinstance(event, tuple):
                    task_obj, _ = event
                    if context_id is None:
                        context_id = task_obj.context_id
                    state = task_obj.status.state.value

                    # Status updates
                    if task_obj.status.message:
                        for p in task_obj.status.message.parts:
                            if isinstance(p.root, TextPart) and p.root.text.strip():
                                print(f"  [status] {p.root.text.strip()}")

                    # Artifacts
                    if task_obj.artifacts:
                        for artifact in task_obj.artifacts:
                            for p in artifact.parts:
                                if isinstance(p.root, TextPart):
                                    last_text = p.root.text
                                elif isinstance(p.root, DataPart):
                                    last_data = p.root.data
                                    if "final_answer" in last_data:
                                        final_answer = last_data["final_answer"]

            # Did we get a FINAL_ANSWER?
            if final_answer is not None:
                print(f"\n{'='*60}")
                print("FINAL ANSWER RECEIVED:")
                print(json.dumps(final_answer, indent=2))
                print(f"{'='*60}")
                break

            # Parse the tool call from purple agent's text response
            action = parse_action(last_text) if last_text else None

            if not action:
                print(f"[<- purple] (no <json> tag found)\n{last_text[:300]}")
                print("[sim] No valid tool call. Ending simulation.")
                break

            tool_name = action.get("name", "")
            tool_args = action.get("arguments", {})
            print(f"[<- purple] tool={tool_name}  args={json.dumps(tool_args)[:120]}")

            if tool_name == "FINAL_ANSWER":
                final_answer = tool_args.get("content", {})
                print(f"\n{'='*60}")
                print("FINAL ANSWER (from text):")
                print(json.dumps(final_answer, indent=2))
                print(f"{'='*60}")
                break

            # Execute tool and prepare next message
            tool_result = execute_tool(tool_name, tool_args, workspace)
            print(f"[sim exec] {tool_result[:200]}{'...' if len(tool_result)>200 else ''}")

            current_message = tool_result

        else:
            print("[sim] Hit MAX_TURNS limit")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def wait_for_server(timeout=15) -> bool:
    for _ in range(timeout * 2):
        time.sleep(0.5)
        try:
            urllib.request.urlopen(f"{BASE_URL}/.well-known/agent.json", timeout=1)
            return True
        except Exception:
            continue
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=list(TASK_TEMPLATES.keys()), default="easy")
    parser.add_argument("--no-server", action="store_true", help="Server already running")
    args = parser.parse_args()

    server = None
    if not args.no_server:
        server = subprocess.Popen(
            [sys.executable, "src/server.py", "--port", "9099"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if not wait_for_server():
            print("ERROR: server did not start")
            server.kill()
            sys.exit(1)
        print(f"[sim] Server started (pid={server.pid})\n")

    try:
        asyncio.run(run_simulation(args.scenario))
    finally:
        if server:
            server.kill()
            server.wait()
            print("\n[sim] Server stopped")


if __name__ == "__main__":
    main()
