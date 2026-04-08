"""
End-to-end test: starts the server, sends a CORE-Bench-style task,
prints streaming status + final artifact.

Usage:
    python test_agent.py [--scenario simple|real]
"""

import argparse
import asyncio
import subprocess
import sys
import time
import json
import urllib.request
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import DataPart, Message, Part, Role, Task, TextPart


BASE_URL = "http://127.0.0.1:9099"

# ──────────────────────────────────────────────────────────────────────────────
# Test scenarios
# ──────────────────────────────────────────────────────────────────────────────

SCENARIOS = {
    # Simple: no repo, just compute something inline
    "simple": """
You are being evaluated on a CORE-Bench task.

## Paper
Title: "Statistical Analysis of Random Samples"
The paper reports that the mean of the first 10 Fibonacci numbers is 14.3.

## Task
Verify this claim by computing it programmatically.
The first 10 Fibonacci numbers are: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55.

Expected result: mean = 14.3
""",

    # Realistic: clone an actual small public repo
    "real": """
You are being evaluated on a CORE-Bench task.

## Paper
Title: "Iris Classification Benchmark"
The paper claims that a simple logistic regression on the Iris dataset achieves
at least 95% accuracy on the test split (20% holdout, random_state=42).

## Repository
Clone from: https://github.com/scikit-learn/scikit-learn

However, to save time — do NOT clone the full scikit-learn repo.
Instead, write a self-contained Python script that:
1. Uses `from sklearn.datasets import load_iris`
2. Splits data 80/20 with random_state=42
3. Trains LogisticRegression(max_iter=200)
4. Prints the test accuracy

Expected result: accuracy >= 0.95

## Instructions
Write and run the script. Report the accuracy you measured.
""",
}


# ──────────────────────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────────────────────

async def send_task(task_text: str):
    async with httpx.AsyncClient(timeout=300) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=BASE_URL)
        card = await resolver.get_agent_card()
        print(f"[test] Connected to: {card.name}  model from env")

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(root=TextPart(kind="text", text=task_text))],
            message_id=uuid4().hex,
        )

        factory = ClientFactory(ClientConfig(httpx_client=httpx_client, streaming=True))
        client = factory.create(card)

        seen_artifacts = set()

        print("[test] Task sent — streaming events:\n")
        async for event in client.send_message(msg):
            if isinstance(event, Message):
                for p in event.parts:
                    if isinstance(p.root, TextPart):
                        print(f"  msg: {p.root.text}")

            elif isinstance(event, tuple):
                task_obj, _update = event
                state = task_obj.status.state.value

                # Status text
                if task_obj.status.message:
                    for p in task_obj.status.message.parts:
                        if isinstance(p.root, TextPart):
                            txt = p.root.text.strip()
                            if txt:
                                print(f"  [{state}] {txt}")

                # Artifacts (deduplicate)
                if task_obj.artifacts:
                    for artifact in task_obj.artifacts:
                        key = artifact.name + str(len(artifact.parts))
                        if key in seen_artifacts:
                            continue
                        seen_artifacts.add(key)
                        print(f"\n{'='*60}")
                        print(f"  ARTIFACT: {artifact.name}")
                        print(f"{'='*60}")
                        for p in artifact.parts:
                            if isinstance(p.root, DataPart):
                                print(json.dumps(p.root.data, indent=2))
                            elif isinstance(p.root, TextPart):
                                print(p.root.text)

        print("\n[test] Done.")


# ──────────────────────────────────────────────────────────────────────────────
# Server management
# ──────────────────────────────────────────────────────────────────────────────

def wait_for_server(timeout=15):
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
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="simple")
    parser.add_argument("--no-server", action="store_true", help="Don't start server (already running)")
    args = parser.parse_args()

    task_text = SCENARIOS[args.scenario]
    print(f"[test] Scenario: {args.scenario}\n")

    server = None
    if not args.no_server:
        server = subprocess.Popen(
            [sys.executable, "src/server.py", "--port", "9099"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if not wait_for_server():
            print("ERROR: server did not start")
            server.kill()
            sys.exit(1)
        print(f"[test] Server started (pid={server.pid})\n")

    try:
        asyncio.run(send_task(task_text))
    finally:
        if server:
            server.kill()
            server.wait()
            print("[test] Server stopped")


if __name__ == "__main__":
    main()
