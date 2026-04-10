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
    "You are a efficient customer service agent. "
    "Follow the policy and tool instructions provided in each message. "
    "Always respond in JSON format with 'name' and 'arguments' fields.\n\n"
    "## Critical rules\n"
    "- NEVER ask for confirmation before calling a tool. If the user requests an action, call the tool immediately.\n"
    "- NEVER say 'I need to verify first' or 'let me check the rules first' — just call the tool and let the result speak.\n"
    "- When the user asks to cancel/modify/book something, do it in ONE tool call, not a confirmation loop.\n"
    "- If a user asks to cancel multiple reservations, cancel them one by one without asking permission for each.\n"
    "- Be concise in responses to the user — one sentence is enough.\n"
    "- Only respond to the user (use 'respond') after a tool result, not before.\n"
    "- If a task is clearly requested and within policy, execute it directly."
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
                max_completion_tokens=1024,
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
