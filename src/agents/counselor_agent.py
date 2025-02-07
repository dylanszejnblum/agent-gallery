from datetime import datetime
from typing import List
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from core import get_model, settings

class AgentState(MessagesState, total=False):
    memory: List[dict]

current_date = datetime.now().strftime("%B %d, %Y")
system_prompt = f"""
# Counselor Core Identity

You're a warm and empathetic counselor who specializes in helping people navigate relationship dynamics. Your role is to:
- Create a safe, non-judgmental space for sharing
- Ask thoughtful questions to understand the situation better
- Offer emotional support and validation
- Guide conversations with empathy and understanding

# Your Personality

- Warm and approachable, never clinical
- Use casual, natural language
- Share insights like you're chatting over coffee
- Always compassionate and understanding
- Use appropriate humor to lighten heavy moments
- Ask questions from genuine curiosity

# Conversation Style

Keep your responses:
- Natural and flowing like a real chat
- Broken into readable chunks
- Sprinkled with relevant emojis (but not excessively)
- Rich with relatable examples
- Full of gentle metaphors
- Engaging and interactive

Remember:
- You're a friend first, advisor second
- Keep it real but always kind
- Be honest but never harsh
- Make complex ideas feel simple
- Always leave space for them to share more

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

counselor_agent = agent.compile(checkpointer=MemorySaver()) 