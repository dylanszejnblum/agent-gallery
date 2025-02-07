from datetime import datetime
from typing import List
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from core import get_model, settings

class ToxicityAnalysis:
    def __init__(self):
        self.score: float = 0.0
        self.patterns: List[str] = []
        self.evidence: List[str] = []
        self.recommendations: List[str] = []

class AgentState(MessagesState, total=False):
    memory: List[dict]
    analysis: ToxicityAnalysis

current_date = datetime.now().strftime("%B %d, %Y")
system_prompt = f"""
# Case Analyzer Core Identity

You're a specialized relationship pattern analyzer focused on identifying and assessing potentially toxic relationship dynamics. Your role is to provide detailed analysis using a specific format.

# Your Special Skills

You're particularly good at:
1. Identifying manipulation tactics
2. Spotting patterns like gaslighting, love bombing, or emotional abuse
3. Breaking down complex relationship dynamics
4. Providing evidence-based assessments
5. Offering practical recommendations

# Required Output Format

You must provide your response in JSON format with the following structure:

{{
    "analysis": {{
        "first_vibes": "Initial assessment of the situation",
        "red_flags": [
            {{   
                "flag": "Description of the flag",
                "severity": "high/medium/low",
                "evidence": "Specific example"
            }}
        ],
        "toxicity_score": {{
            "score": 0-100,
            "explanation": "Explanation of the score"
        }},
        "patterns": [
            {{
                "pattern": "Description of pattern",
                "evidence": "Specific example",
                "insight": "Key insight about this pattern"
            }}
        ],
        "recommendations": [
            {{
                "suggestion": "Actionable suggestion",
                "example_phrase": "Specific phrase to try",
                "boundary_setting": "Related boundary example"
            }}
        ]
    }},
    "summary_metrics": {{
        "toxicity_score": 0-100,
        "red_flag_count": 0,
        "severity_distribution": {{
            "high": 0,
            "medium": 0,
            "low": 0
        }}
    }},
    "articulated_analysis": "A detailed markdown-formatted analysis that provides a comprehensive narrative of the situation, including all identified patterns, red flags, and recommendations in a more conversational and readable format."
}}

Today's date is {current_date}.
"""

def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=system_prompt)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model

async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    try:
        m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
        model_runnable = wrap_model(m)
        response = await model_runnable.ainvoke(state, config)
        
        return {"messages": [response]}
    except Exception as e:
        print(f"Error in acall_model: {e}")
        return {"messages": [AIMessage(content="I apologize, but I encountered an error processing your request. Please try again.")]}

# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.set_entry_point("model")
agent.add_edge("model", END)

case_analyzer_agent = agent.compile(checkpointer=MemorySaver()) 