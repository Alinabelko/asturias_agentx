import argparse
import os
import sys

# Allow imports from src/ when running directly
sys.path.insert(0, os.path.dirname(__file__))

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="CORE-Bench research reproduction purple agent")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9019)
    parser.add_argument("--card-url", type=str, help="Public URL for agent card")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use")
    args = parser.parse_args()

    os.environ.setdefault("AGENT_MODEL", args.model)

    skill = AgentSkill(
        id="research_reproduction",
        name="Research Reproduction",
        description=(
            "Reproduces computational results from scientific papers. "
            "Given a paper and its associated code/data repository, "
            "the agent executes the code and verifies that results match the paper's claims."
        ),
        tags=["research", "reproducibility", "core-bench", "science"],
        examples=[],
    )

    agent_card = AgentCard(
        name="research_reproduction_agent",
        description="CORE-Bench agent: reproduces scientific paper results from code and data",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text", "data"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    print(f"Starting research agent on {args.host}:{args.port} | model={os.environ['AGENT_MODEL']}")
    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        timeout_keep_alive=600,
    )


if __name__ == "__main__":
    main()
