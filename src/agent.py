"""
tau2-bench Purple Agent.

Protocol:
  Each A2A turn = one user message (or tool result) from the green agent.
  Agent responds with JSON artifact:
    {"name": "tool_name", "arguments": {...}}   — tool call
    {"name": "respond",   "arguments": {"content": "..."}}  — reply to user

  The agent maintains conversation history across turns (same context_id).
"""

import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful customer service agent. "
    "Follow the policy and tool instructions provided in each message. "
    "Always respond in JSON format with 'name' and 'arguments' fields."
)


class ResearchAgent:
    def __init__(self):
        self.model = os.getenv("AGENT_MODEL", "gpt-4o-mini")
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(TaskState.working, new_agent_text_message("Thinking..."))

        self.messages.append({"role": "user", "content": input_text})

        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=1024,
            )
            content = completion.choices[0].message.content or "{}"
            result = json.loads(content)
        except Exception as e:
            result = {"name": "respond", "arguments": {"content": f"Error: {e}"}}
            content = json.dumps(result)

        self.messages.append({"role": "assistant", "content": content})

        await updater.add_artifact(
            parts=[Part(root=DataPart(data=result))],
            name="Action",
        )
