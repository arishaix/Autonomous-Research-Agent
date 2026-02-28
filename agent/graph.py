from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import planner_node, research_node, writer_node, critic_node

def route_after_critic(state: AgentState) -> Literal["planner", "END"]:
    """
    Conditional edge routing logic.
    If the critic left revision notes, we route back to the planner to generate new searches.
    Otherwise, we route to END to finish the graph.
    """
    if state.get("revision_notes"):
        print("\n--- CRITIC REJECTED DRAFT. LOOPING BACK TO PLANNER ---\n")
        return "planner"
    else:
        print("\n--- CRITIC APPROVED DRAFT. FINISHING ---\n")
        return "END"

def build_graph():
    """
    Constructs the LangGraph by adding nodes and connecting them with edges.
    """
    # 1. Initialize the StateGraph with our custom State schema
    workflow = StateGraph(AgentState)
    
    # 2. Add our nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", research_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    
    # 3. Add regular edges (the linear flow)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "critic")
    
    # 4. Add the conditional edge
    # This checks the output of the critic node.
    workflow.add_conditional_edges(
        "critic",                 # The node we are coming from
        route_after_critic,       # The function that decides where to go
        {
            "planner": "planner", # If it returns 'planner', go to 'planner'
            "END": END            # If it returns 'END', go to END
        }
    )
    
    # 5. Compile the graph into a runnable application
    app = workflow.compile()
    
    return app
