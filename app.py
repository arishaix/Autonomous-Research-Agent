import streamlit as st
import os
from agent.graph import build_graph

st.set_page_config(page_title="Autonomous Research Agent", page_icon="🕵️", layout="wide")

st.title("🕵️ Autonomous Research Agent")
st.markdown("Enter a research topic below. The agent will autonomously plan searches, scrape the web, draft a report, and critique its own work until it meets a high standard.")

# Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. **Planner**: Breaks your topic into 1-3 specific Google searches.
    2. **Researcher**: Searches the web and scrapes the top pages.
    3. **Writer**: Writes a factual draft based on the scraped content.
    4. **Critic**: Evaluates the draft. If weak, loops back to the Planner for more research.
    """)

query = st.text_input("What would you like me to research?", "Research how AI agents are used in fintech")

if st.button("Start Research"):
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("Please enter your OpenAI API Key in the sidebar.")
    else:
        st.info("Agent is thinking... Check the logs below for intermediate steps.")
        
        # Build the graph
        app = build_graph()
        
        # Initial state
        initial_state = {
            "query": query,
            "search_plan": [],
            "documents": [],
            "draft": "",
            "revision_notes": "",
            "loop_count": 0
        }
        
        # We will stream the output of each node to the frontend
        with st.status("Agent execution log", expanded=True) as status:
            try:
                # Accumulate state as it streams
                current_state = initial_state.copy()
                
                # Stream the state updates
                for output in app.stream(initial_state):
                    # output is a dict where the key is the node name and the value is the state update
                    for node_name, state_update in output.items():
                        if node_name == "planner":
                            st.write(f"🧩 **Planner** generated search plan: {state_update.get('search_plan')}")
                            current_state.update(state_update)
                        elif node_name == "researcher":
                            docs = state_update.get("documents", [])
                            st.write(f"🌐 **Researcher** retrieved {len(docs)} documents.")
                            # The schema uses operator.add, so we manually append to our local state
                            current_state["documents"].extend(docs)
                        elif node_name == "writer":
                            st.write("📝 **Writer** drafted a section of the report.")
                            current_state.update(state_update)
                        elif node_name == "critic":
                            notes = state_update.get("revision_notes", "")
                            if notes:
                                st.warning(f"🧐 **Critic** rejected the draft. Reasons: {notes}")
                            else:
                                st.success("✅ **Critic** approved the draft!")
                            current_state.update(state_update)
                status.update(label="Research Complete!", state="complete", expanded=False)
                
                st.subheader("Final Research Report")
                st.markdown(current_state.get("draft", "No draft generated."))
                
                with st.expander("View Retrieved Sources"):
                    for doc in current_state.get("documents", []):
                        st.markdown(f"- [{doc['title']}]({doc['url']})")
                        
            except Exception as e:
                status.update(label="Error occurred", state="error", expanded=True)
                st.error(f"An error occurred: {e}")
