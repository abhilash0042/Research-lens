import streamlit as st
import os
import re
import tempfile
import pandas as pd
from typing import List

# Import our backend
from src.ingestion import ingest_paper
from src.indexer import load_index
from src.pipeline import ask_question
from src.intelligence import (
    summarize_paper, 
    detect_contradictions, 
    generate_comparison_table, 
    generate_literature_review,
    generate_hypotheses
)
from src.utils import UnifiedIndex, PaperResult

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="ResearchLens",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 ResearchLens")
st.markdown("*Advanced Multi-Paper AI Research Assistant*")

# --- Session State ---
if "unified_indices" not in st.session_state:
    st.session_state.unified_indices = []
if "paper_results" not in st.session_state:
    st.session_state.paper_results = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
# State for generated results
if "generated_summaries" not in st.session_state:
    st.session_state.generated_summaries = {}
if "comparison_table" not in st.session_state:
    st.session_state.comparison_table = None
if "contradictions" not in st.session_state:
    st.session_state.contradictions = None
if "lit_review" not in st.session_state:
    st.session_state.lit_review = None
if "hypotheses" not in st.session_state:
    st.session_state.hypotheses = None


# --- Helper: Render Interactive Citations ---
def render_interactive_text(text: str):
    """
    Finds [SOURCE N: Title, Section] and renders it beautifully.
    """
    styled_text = re.sub(
        r'(\[SOURCE \d+:.*?\])', 
        r'<span style="background-color: #2e3b4e; color: #a3c2f0; padding: 2px 6px; border-radius: 4px; font-size: 0.85em;">\1</span>', 
        text
    )
    st.markdown(styled_text, unsafe_allow_html=True)


# --- Sidebar: Upload & Manage PDFs ---
with st.sidebar:
    st.header("📄 Knowledge Base")
    uploaded_files = st.file_uploader("Upload Academic Papers (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Ingest Papers"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing & embedding papers..."):
                for uploaded_file in uploaded_files:
                    if uploaded_file.name in st.session_state.processed_files:
                        continue
                        
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        st.info(f"Ingesting: {uploaded_file.name}...")
                        result = ingest_paper(tmp_path)
                        
                        if result:
                            # Load index to verify
                            unified, _, _ = load_index(result.paper_id)
                            st.session_state.unified_indices.append(unified)
                            st.session_state.paper_results.append(result)
                            st.session_state.processed_files.add(uploaded_file.name)
                            st.success(f"Successfully processed {uploaded_file.name}")
                        else:
                            st.error(f"Skipped {uploaded_file.name}: Not enough text.")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                    finally:
                        os.unlink(tmp_path)
    
    st.divider()
    st.subheader("📚 Loaded Papers")
    if not st.session_state.paper_results:
        st.caption("No papers loaded yet.")
    else:
        for pr in st.session_state.paper_results:
            st.markdown(f"- **{pr.metadata.title}** ({pr.metadata.year})")
            
    if st.button("Clear Memory"):
        st.session_state.unified_indices = []
        st.session_state.paper_results = []
        st.session_state.chat_history = []
        st.session_state.processed_files = set()
        st.session_state.generated_summaries = {}
        st.session_state.comparison_table = None
        st.session_state.contradictions = None
        st.session_state.lit_review = None
        st.session_state.hypotheses = None
        st.rerun()


# --- Main Application Tabs ---
tab1, tab2, tab3 = st.tabs(["💬 Chat & QA", "📑 Summaries", "🧠 Multi-Paper Intelligence"])

# --- Tab 1: Chat & RAG ---
with tab1:
    st.subheader("Chat with your Knowledge Base")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_interactive_text(message["content"])
            else:
                st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question across all papers... (e.g. 'What is the methodology used?')"):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            if not st.session_state.unified_indices:
                st.warning("Please upload and ingest papers first.")
            else:
                with st.spinner("Searching and Reasoning..."):
                    answer = ask_question(prompt, st.session_state.unified_indices, st.session_state.chat_history)
                    render_interactive_text(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})


# --- Tab 2: Summaries ---
with tab2:
    st.subheader("Deep Paper Summaries")
    if not st.session_state.paper_results:
        st.info("Upload papers to generate summaries.")
    else:
        # Create a dropdown to select which paper to summarize
        titles = [pr.metadata.title for pr in st.session_state.paper_results]
        selected_title = st.selectbox("Select a paper to summarize:", titles)
        
        # Display existing summary if it exists
        if selected_title in st.session_state.generated_summaries:
            summary = st.session_state.generated_summaries[selected_title]
        else:
            summary = None
            
        if st.button("Generate Detailed Summary"):
            selected_pr = next(pr for pr in st.session_state.paper_results if pr.metadata.title == selected_title)
            with st.spinner("Generating highly detailed structured summary using Llama-3.1..."):
                summary = summarize_paper(selected_pr)
                st.session_state.generated_summaries[selected_title] = summary
                
        if summary:
            st.markdown(f"### {summary.title}")
            
            with st.expander("🎯 Core Contribution", expanded=True):
                st.write(summary.contribution)
                
            with st.expander("🔬 Methodology", expanded=True):
                st.write(summary.methodology)
                
            with st.expander("📊 Results", expanded=True):
                st.write(summary.results)
                
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Datasets Used:**")
                st.info(summary.datasets)
            with col2:
                st.markdown("**Limitations & Future Work:**")
                st.warning(summary.limitations)


# --- Tab 3: Intelligence ---
with tab3:
    st.subheader("Cross-Paper AI Intelligence")
    
    if len(st.session_state.paper_results) < 2:
        st.info("Upload at least TWO papers to unlock cross-paper intelligence.")
    else:
        st.markdown("Use these advanced tools to synthesize knowledge across all loaded papers.")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Action 1: Comparison Table
        if col1.button("📊 Compare Papers", use_container_width=True):
            with st.spinner("Analyzing dimensions..."):
                rows = generate_comparison_table(st.session_state.paper_results)
                st.session_state.comparison_table = rows
                
        if st.session_state.comparison_table:
            rows = st.session_state.comparison_table
            data = {}
            for row in rows:
                data[row.dimension] = row.values
            df = pd.DataFrame(data).T
            st.dataframe(df, use_container_width=True)
                    
        # Action 2: Contradictions
        if col2.button("⚔️ Find Contradictions", use_container_width=True):
            with st.spinner("Detecting conflicting claims..."):
                contradictions = detect_contradictions(st.session_state.paper_results)
                st.session_state.contradictions = contradictions if contradictions else "None"
                
        if st.session_state.contradictions:
            if st.session_state.contradictions == "None":
                st.success("No major contradictions found among the papers.")
            else:
                contradictions = st.session_state.contradictions
                st.error(f"Found {len(contradictions)} conflicting claims across the literature.")
                for i, c in enumerate(contradictions):
                    with st.expander(f"Contradiction {i+1}: {c.paper_a} vs {c.paper_b}"):
                        st.markdown(f"**{c.paper_a} claims:** {c.claim_a}")
                        st.markdown(f"**{c.paper_b} claims:** {c.claim_b}")
                        st.info(f"**AI Analysis:** {c.explanation}")

        # Action 3: Literature Review
        if col3.button("📝 Write Lit Review", use_container_width=True):
            with st.spinner("Synthesizing multi-cited review..."):
                lit_review = generate_literature_review(st.session_state.paper_results)
                st.session_state.lit_review = lit_review
                
        if st.session_state.lit_review:
            st.markdown("### Synthesized Literature Review")
            render_interactive_text(st.session_state.lit_review)
                
        # Action 4: Auto-Hypotheses (Advanced Feature)
        if col4.button("💡 Generate Hypotheses", use_container_width=True):
            with st.spinner("Finding gaps and generating novel research ideas..."):
                hypotheses = generate_hypotheses(st.session_state.paper_results)
                st.session_state.hypotheses = hypotheses
                
        if st.session_state.hypotheses:
            st.markdown("### Novel Research Hypotheses")
            st.markdown(st.session_state.hypotheses)
