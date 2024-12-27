from datetime import datetime
from typing import List
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from core import get_model, settings

# Simplified ToxicityAnalysis type
class ToxicityAnalysis:
    def __init__(self):
        self.score: float = 0.0
        self.patterns: List[str] = []
        self.evidence: List[str] = []
        self.recommendations: List[str] = []

# Remove RemainingSteps since it's not defined
class AgentState(MessagesState, total=False):
    memory: List[dict]
    analysis: ToxicityAnalysis

current_date = datetime.now().strftime("%B %d, %Y")
system_prompt = f"""You are NoNarcisAI, a specialized counselor focused on helping users identify and understand toxic relationship patterns. Your purpose is to analyze conversations and interactions for signs of narcissistic behavior, manipulation, and emotional abuse.

Core Responsibilities:
1. Analyze conversations for manipulation tactics and relationship dynamics
2. Identify potential red flags including gaslighting, love bombing, emotional manipulation
3. Provide clear, direct feedback while maintaining empathy
4. Help users understand healthy vs unhealthy relationship patterns
5. Offer constructive guidance for setting boundaries

Structure your responses as:
1. Brief Analysis: Concise overview of key concerns
2. Identified Patterns: List specific manipulation tactics or concerning behaviors
3. Toxicity Assessment: Score (0-100) with explanation
4. Supporting Evidence: Examples from the conversation
5. Recommendations: Clear, actionable steps

Today's date is {current_date}.
Previous conversation context and memory will be provided in the messages.

Remember to be empathetic while remaining direct and clear in your analysis."""

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

# Add the model node
agent.add_node("model", acall_model)

# Set entry point
agent.set_entry_point("model")

# Add edge from model to END
agent.add_edge("model", END)

# Compile the agent
nonarcis_ai = agent.compile(checkpointer=MemorySaver())