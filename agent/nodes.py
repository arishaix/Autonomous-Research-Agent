from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os

from agent.state import AgentState
from tools.search import perform_search, scrape_webpage

# Define the structured output we want from the LLM
class SearchPlan(BaseModel):
    queries: List[str] = Field(
        description="A list of 1 to 3 distinct search queries to research the user's prompt."
    )

def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    The Planner Node.
    Takes the user's query and breaks it down into specific search terms.
    If we are looping (revision_notes is present), it adjusts the queries based on the Critic's feedback.
    """
    query = state.get("query", "")
    revision_notes = state.get("revision_notes", "")
    
    # Initialize the LLM. We are using OpenAI for this example.
    # Make sure OPENAI_API_KEY is set in your environment variables.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # We bind our pydantic model to the LLM so it strictly outputs the list of strings
    structured_llm = llm.with_structured_output(SearchPlan)
    
    # Default system prompt
    system_prompt = (
        "You are an expert research planner. Your job is to take a user's broad research topic "
        "and break it down into 1 to 3 highly specific Google search queries that will yield "
        "the most relevant and factual information."
    )
    
    # If the Critic node gave us feedback from a previous run, adjust the plan!
    if revision_notes:
        system_prompt += (
            f"\n\nWe already did an initial search, but our draft was rejected by the reviewer. "
            f"Here is why: '{revision_notes}'.\n"
            f"Generate NEW search queries to fill in these missing gaps."
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Research Topic: {query}")
    ]
    
    # Call the LLM
    response = structured_llm.invoke(messages)
    
    # Update the state. 
    # Because we return a dict with the key "search_plan", LangGraph will update 
    # the AgentState with this new list of queries.
    return {"search_plan": response.queries}

def research_node(state: AgentState) -> Dict[str, Any]:
    """
    The Research Node (Combines Search and Retriever).
    Takes the search_plan, executes the searches, and retrieves the raw text
    for the top results of each query.
    """
    search_plan = state.get("search_plan", [])
    documents = state.get("documents", [])
    
    new_documents = []
    seen_urls = set()
    
    # We do not want to rescrape urls if we have already scraped them in previous loops
    for doc in documents:
        seen_urls.add(doc.get("url"))
    
    for query in search_plan:
        print(f"  -> Searching web for: '{query}'")
        search_results = perform_search(query, max_results=2)
        
        for result in search_results:
            url = result["url"]
            if url not in seen_urls:
                seen_urls.add(url)
                print(f"  -> Scraping: {url}")
                content = scrape_webpage(url)
                
                if content:
                    new_documents.append({
                        "query": query,
                        "url": url,
                        "title": result["title"],
                        "content": content
                    })
                    
    # State update: 'documents' uses operator.add, so these new_documents 
    # will be APPENDED to the existing documents in the state list by LangGraph.
    return {"documents": new_documents}

def writer_node(state: AgentState) -> Dict[str, Any]:
    """
    The Writer Node.
    Takes the original query and the retrieved documents, and generates a draft report.
    If it's a revision, it also considers the Critic's notes.
    """
    query = state.get("query", "")
    documents = state.get("documents", [])
    revision_notes = state.get("revision_notes", "")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    # Format the retrieved documents into a single string for the prompt
    context = ""
    for i, doc in enumerate(documents):
        context += f"[{i+1}] Source: {doc['url']}\nContent: {doc['content'][:1000]}...\n\n"
        
    system_prompt = (
        "You are an expert research analyst. Your job is to write a comprehensive, factual "
        "report answering the user's query based ONLY on the provided source documents. "
        "Always cite your sources using the [number] format in the text."
    )
    
    user_prompt = f"Query: {query}\n\nSources:\n{context}"
    
    # If the Critic node gave us feedback from a previous run, adjust the draft!
    if revision_notes:
        user_prompt += (
            f"\n\n--- REVISION NEEDED ---\n"
            f"Your previous draft was reviewed and received this feedback: '{revision_notes}'.\n"
            f"Please rewrite the report to address this feedback."
        )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Update the state with the newly generated draft report
    return {"draft": response.content}

class CriticFeedback(BaseModel):
    is_acceptable: bool = Field(
        description="True if the report adequately answers the query with facts. False if it is too brief, misses the point, or lacks concrete details."
    )
    feedback_notes: str = Field(
        description="If not acceptable, provide specific instructions on what is missing or what needs to be researched further. If acceptable, return an empty string."
    )

def critic_node(state: AgentState) -> Dict[str, Any]:
    """
    The Critic Node.
    Evaluates the draft. If it's good, clears revision_notes. 
    If it's bad, populates revision_notes so the graph knows to loop back to Search.
    """
    query = state.get("query", "")
    draft = state.get("draft", "")
    loop_count = state.get("loop_count", 0)
    
    # Base case to prevent infinite loops (cut off after 3 tries)
    if loop_count >= 2:
        return {"revision_notes": ""}
        
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(CriticFeedback)
    
    system_prompt = (
        "You are a strict research reviewer. Your job is to evaluate the provided draft report "
        "to ensure it comprehensively and factually answers the user's query.\n"
        "If it is good, set is_acceptable to True.\n"
        "If it is vague, too short, or misses the core question, set is_acceptable to False "
        "and provide specific feedback on what else needs to be researched."
    )
    
    user_prompt = f"Original Query: {query}\n\nDraft Report:\n{draft}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = structured_llm.invoke(messages)
    
    # Update state.
    # Increment loop count so we don't get stuck forever.
    # If acceptable, clear the notes (an empty string means the graph will End).
    # If not acceptable, pass the notes (a populated string means the graph will loop to Planner/Search).
    return {
        "revision_notes": "" if response.is_acceptable else response.feedback_notes,
        "loop_count": loop_count + 1
    }
