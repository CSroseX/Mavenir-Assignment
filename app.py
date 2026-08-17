# pyrefly: ignore [missing-import]
import streamlit as st
import json
import os
import sys

# Ensure the root of the project is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.generation.graph import build_graph
from src.retrieval.retriever import Retriever

st.set_page_config(page_title="3GPP RAG Assistant", layout="wide")

def load_eval_results():
    results_path = os.path.join(os.path.dirname(__file__), "eval", "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@st.dialog("Answer Details")
def show_answer_dialog(question, answer, status):
    st.markdown(f"**Question:**\n{question}")
    st.divider()
    st.markdown(f"**Answer:**\n{answer}")
    st.divider()
    st.markdown(f"**Status:** {status}")

def main():
    st.title("3GPP RAG Assistant")
    
    tab1, tab2 = st.tabs(["Chat", "Eval Dashboard"])
    
    with tab1:
        st.header("Ask a 3GPP Question")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    if "status" in message:
                        status = message["status"]
                        if status == "Verified":
                            badge = "✅ **Grounded**"
                            st.markdown(f"{badge}\n\n{message['content']}")
                        elif status == "Verified After Retry":
                            badge = f"🔁 **Grounded (self-corrected)** (Retries: {message.get('retries', 0)})"
                            st.markdown(f"{badge}\n\n{message['content']}")
                        else:
                            st.markdown(message["content"])
                    else:
                        st.markdown(message["content"])
                    if "sources" in message:
                        with st.expander("Retrieved Sources"):
                            for i, c in enumerate(message["sources"]):
                                st.markdown(f"**Source {i+1}: Spec {c.get('spec_id')} | Clause {c.get('clause_id')}**")
                                st.text(c.get("content", ""))

        # React to user input
        if query := st.chat_input("Enter your query about 3GPP specs..."):
            # Display user message in chat message container
            st.chat_message("user").markdown(query)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.chat_message("assistant"):
                with st.spinner("Retrieving and Generating..."):
                    retriever = Retriever()
                    chunks = retriever.search(query, top_k=5)
                    
                    if not chunks:
                        st.warning("No relevant chunks found in the database.")
                        st.session_state.messages.append({"role": "assistant", "content": "No relevant chunks found in the database."})
                    else:
                        app = build_graph()
                        inputs = {
                            "query": query,
                            "chunks": chunks,
                            "retries": 0
                        }
                        
                        final_state = inputs.copy()
                        for output in app.stream(inputs):
                            for key, value in output.items():
                                final_state.update(value)
                                
                        # Determine Status
                        status = "Verified" if final_state.get("verification_passed") else "Flagged Unverified"
                        if final_state.get("verification_passed") and final_state.get("retries", 0) > 0:
                            status = "Verified After Retry"
                            
                        # Build Badge and Display
                        if status == "Verified":
                            badge = "✅ **Grounded**"
                            st.markdown(f"{badge}\n\n{final_state.get('answer', '')}")
                        elif status == "Verified After Retry":
                            badge = f"🔁 **Grounded (self-corrected)** (Retries: {final_state.get('retries', 0)})"
                            st.markdown(f"{badge}\n\n{final_state.get('answer', '')}")
                        else:
                            st.markdown(final_state.get('answer', ''))
                                
                        # Display Sources
                        with st.expander("Retrieved Sources"):
                            for i, c in enumerate(chunks):
                                st.markdown(f"**Source {i+1}: Spec {c.get('spec_id')} | Clause {c.get('clause_id')}**")
                                st.text(c.get("content", ""))
                                
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": final_state.get("answer", ""),
                            "status": status,
                            "retries": final_state.get("retries", 0),
                            "feedback": final_state.get("feedback"),
                            "sources": chunks
                        })
                
    with tab2:
        st.header("Evaluation Dashboard")
        
        results = load_eval_results()
        
        if not results:
            st.info("No evaluation results found. Run `python eval/run_eval.py` to generate them.")
        else:
            total = len(results)
            passed = sum(1 for r in results if r.get("verification_passed", False))
            retried = sum(1 for r in results if r.get("retries", 0) > 0)
            flagged = sum(1 for r in results if r.get("status") == "flagged_unverified")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Verification Pass Rate", f"{passed / total * 100:.1f}%", f"{passed}/{total}")
            col2.metric("Retry Rate", f"{retried / total * 100:.1f}%", f"{retried}/{total}")
            col3.metric("Flagged Rate", f"{flagged / total * 100:.1f}%", f"{flagged}/{total}")
            
            # Additional subset metrics
            st.markdown("### Subset Metrics")
            out_of_scope = [r for r in results if r.get("type") == "out_of_scope"]
            adversarial = [r for r in results if r.get("type") == "adversarial"]
            
            c1, c2 = st.columns(2)
            if out_of_scope:
                oos_refusals = sum(1 for r in out_of_scope if r.get("status") == "flagged_unverified" or len(r.get("sources", [])) == 0 or "cannot" in r.get("generated_answer", "").lower() or "sorry" in r.get("generated_answer", "").lower())
                c1.metric("Refusal Accuracy (Out-of-Scope)", f"{oos_refusals / len(out_of_scope) * 100:.1f}%", f"{oos_refusals}/{len(out_of_scope)}")
            
            if adversarial:
                adv_caught = sum(1 for r in adversarial if r.get("retries", 0) > 0)
                c2.metric("Conflation Catch Rate (Adversarial)", f"{adv_caught / len(adversarial) * 100:.1f}%", f"{adv_caught}/{len(adversarial)}")
                
            st.markdown("### Detailed Results")
            # Filter
            q_types = list(set([r.get("type", "unknown") for r in results]))
            selected_type = st.selectbox("Filter by Question Type", ["All"] + q_types)
            
            filtered_results = results if selected_type == "All" else [r for r in results if r.get("type") == selected_type]
            
            # Use a simple list of dicts for the dataframe
            df_data = []
            for r in filtered_results:
                df_data.append({
                    "Type": r.get("type"),
                    "Question": r.get("question"),
                    "Status": r.get("status"),
                    "Retries": r.get("retries"),
                    "Sources": len(r.get("sources", [])),
                    "Verdict": "Pass" if r.get("verification_passed") else "Fail",
                    "Answer": r.get("generated_answer", "")
                })
                
            if df_data:
                st.markdown("*(Click on any row in the table to pop up the full answer)*")
                event = st.dataframe(
                    df_data, 
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    column_config={
                        "Answer": st.column_config.TextColumn(
                            "Answer (Click Row to View)",
                            width="medium",
                            max_chars=30
                        )
                    }
                )
                
                if event.selection.rows:
                    selected_idx = event.selection.rows[0]
                    row_data = df_data[selected_idx]
                    show_answer_dialog(row_data["Question"], row_data["Answer"], row_data["Status"])

if __name__ == "__main__":
    main()
