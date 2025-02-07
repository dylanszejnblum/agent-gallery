from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from agents.bg_task_agent.bg_task_agent import bg_task_agent
from agents.chatbot import chatbot
from agents.research_assistant import research_assistant
from schema import AgentInfo
from agents.nonarcisai import nonarcis_ai

from agents.counselor_agent import counselor_agent
from agents.case_analyzer_agent import case_analyzer_agent
from agents.npc_agent import npc_agent




DEFAULT_AGENT = "research-assistant"


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "chatbot": Agent(description="A simple chatbot.", graph=chatbot),
    "research-assistant": Agent(
        description="A research assistant with web search and calculator.", graph=research_assistant
    ),
    "bg-task-agent": Agent(description="A background task agent.", graph=bg_task_agent),
    "nonarcis-ai": Agent(
        description="A specialized counselor that analyzes conversations for toxic relationship patterns.",
        graph=nonarcis_ai
    ),
    "counselor": Agent(
        description="A warm and empathetic counselor for relationship discussions.",
        graph=counselor_agent
    ),
    "case-analyzer": Agent(
        description="A specialized analyzer for relationship patterns and toxicity assessment.",
        graph=case_analyzer_agent
    ),
    "npc-agent": Agent(
        description=(
            "An AI character with a specific personality that interacts with users. "
            "Always stays in character, using appropriate tone, style, and emojis as defined."
        ),
        graph=npc_agent
    )
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
