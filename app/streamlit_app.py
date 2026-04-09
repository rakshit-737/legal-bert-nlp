"""
Streamlit UI for Legal BERT NLP Application
Interactive document processing interface
"""
import streamlit as st
import torch
import sys
import os
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from inference.processor import LegalDocumentProcessor, DocumentSummarizer, batch_process_documents
from preprocessing.text_cleaner import TextCleaner


# Page configuration
st.set_page_config(
    page_title="⚖️ Legal BERT NLP",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with premium design aesthetics
st.markdown("""
<style>
    :root {
        --primary: #0F3460;
        --secondary: #533483;
        --accent: #00D9FF;
        --success: #1ABC9C;
        --warning: #F39C12;
        --error: #E74C3C;
        --light: #ECF0F1;
        --dark: #2C3E50;
    }
    
    * {
        font-family: 'Segoe UI', -apple-system, sans-serif;
    }
    
    .main-header {
        font-size: 3.5em;
        font-weight: 700;
        background: linear-gradient(135deg, #0F3460 0%, #533483 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5em;
        letter-spacing: -1px;
    }
    
    .section-header {
        font-size: 2em;
        font-weight: 600;
        color: #0F3460;
        border-bottom: 3px solid #00D9FF;
        padding-bottom: 0.75em;
        margin: 1.5em 0;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2em;
        border-radius: 12px;
        margin: 1.2em 0;
        border: 1px solid rgba(0, 217, 255, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(51, 130, 211, 0.15);
    }
    
    .entity-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #f0e8f8 100%);
        padding: 1.2em;
        border-left: 5px solid #00D9FF;
        margin: 0.8em 0;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .entity-box:hover {
        border-left-color: #533483;
        background: linear-gradient(135deg, #e0ecf6 0%, #ebe0f6 100%);
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1em;
        border-radius: 8px;
        border-left: 5px solid #1ABC9C;
        font-weight: 500;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1em;
        border-radius: 8px;
        border-left: 5px solid #E74C3C;
        font-weight: 500;
    }
    
    .info-message {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1em;
        border-radius: 8px;
        border-left: 5px solid #17a2b8;
        font-weight: 500;
    }
    
    .task-label {
        font-size: 1.2em;
        font-weight: 600;
        color: #0F3460;
        margin: 1.5em 0 0.5em 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load models (cached for performance)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = LegalDocumentProcessor(device=device)
    return processor


def main():
    st.markdown(
        "<h1 class='main-header'>⚖️ Legal BERT NLP Processor</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<div style='text-align: center; font-size: 1.2em; color: #666; margin-bottom: 2em;'>"
        "🚀 Advanced NLP for legal documents | Classification • NER • Similarity • Summarization"
        "</div>",
        unsafe_allow_html=True
    )
    
    # Sidebar
    st.sidebar.markdown("## 🔧 Configuration")
    task = st.sidebar.radio(
        "Select Analysis Task:",
        ["📄 Document Classification", "🔍 Named Entity Recognition",
         "🔗 Similarity Analysis", "📊 Document Summarization",
         "⚡ Batch Processing"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Display Options")
    show_metrics = st.sidebar.checkbox("Show detailed metrics", value=True, help="Display additional analysis metrics")
    show_probabilities = st.sidebar.checkbox("Show confidence scores", value=True, help="Display per-class confidence breakdown")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 System Info")
    st.sidebar.markdown(f"""
    **Model Configuration**
    - Base Model: {config.DEFAULT_MODEL}
    - Framework: PyTorch + Transformers
    - Hardware: {'🚀 GPU (CUDA)' if torch.cuda.is_available() else '💻 CPU'}
    - Classification: {len(config.CLASSIFICATION_LABELS)} document types
    """)
    
    # Load models with better error handling
    try:
        with st.spinner("🔄 Loading AI models..."):
            processor = load_models()
        st.sidebar.success("✅ Models loaded successfully")
    except Exception as e:
        st.sidebar.error(f"⚠️ Model loading error: {str(e)[:60]}")
        st.error("**Unable to load models.** Please ensure all dependencies are installed with: `pip install -r requirements.txt`")
        return
    
    # Task routing
    if task == "📄 Document Classification":
        document_classification(processor, show_metrics, show_probabilities)
    
    elif task == "🔍 Named Entity Recognition":
        named_entity_recognition(processor, show_metrics)
    
    elif task == "🔗 Similarity Analysis":
        similarity_analysis(processor)
    
    elif task == "📊 Document Summarization":
        document_summarization(processor, show_metrics)
    
    elif task == "⚡ Batch Processing":
        batch_processing(processor)


def document_classification(processor, show_metrics, show_probabilities):
    """Document classification interface - identify document type"""
    st.markdown("<h2 class='section-header'>📄 Document Classification</h2>", unsafe_allow_html=True)
    
    st.markdown(
        "<div style='background: #f0f7ff; padding: 1em; border-radius: 8px; margin-bottom: 1.5em; border-left: 4px solid #00D9FF;'>"
        "<strong>💡 What this does:</strong> Analyzes your legal document and classifies it into one of "
        f"{len(config.CLASSIFICATION_LABELS)} predefined document types with confidence scoring."
        "</div>",
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Paste your legal document here:",
            height=280,
            placeholder="Enter the legal document text you want to classify (contracts, court cases, appeals, statutes, etc.)...",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("<div class='task-label'>📋 Supported Types</div>", unsafe_allow_html=True)
        for label, idx in config.CLASSIFICATION_LABELS.items():
            st.markdown(f"• {label}", help=f"Document type: {label}")
    
    if st.button("🔍 Analyze Document", key="classify", use_container_width=True):
        if not text_input.strip():
            st.error(
                "**⚠️ Please provide a document.** Paste or type at least some legal text to analyze. "
                "Need an example? Try: 'This is a contract between Party A and Party B for consulting services.'"
            )
            return
        
        try:
            with st.spinner("🧠 Analyzing document..."):
                result = processor.classify_document(text_input, return_proba=show_probabilities)
            
            # Success banner
            st.markdown(
                f"<div class='success-message'>✅ Analysis Complete</div>",
                unsafe_allow_html=True
            )
            
            # Main classification result
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(
                    f"<div class='metric-box'>"
                    f"<strong style='font-size: 1.1em;'>Document Type</strong><br>"
                    f"<span style='font-size: 2em; color: #0F3460; font-weight: 700;'>{result['label'].upper()}</span><br>"
                    f"<span style='color: #666; font-size: 0.9em;'>Confidence: {result['confidence']:.1%}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            with col2:
                confidence_pct = int(result['confidence'] * 100)
                st.metric(
                    label="Confidence",
                    value=f"{confidence_pct}%",
                    delta="High confidence" if confidence_pct >= 80 else ("Fair confidence" if confidence_pct >= 60 else "Low confidence")
                )
            
            with col3:
                st.metric(
                    label="Analysis",
                    value="Reliable" if confidence_pct >= 80 else ("Moderate" if confidence_pct >= 60 else "Review")
                )
            
            # Detailed confidence breakdown
            if show_probabilities and "all_scores" in result:
                st.markdown("### 📊 Classification Scores for All Types")
                scores_df = pd.DataFrame(
                    [(k, v) for k, v in result["all_scores"].items()],
                    columns=["Document Type", "Confidence"]
                ).sort_values("Confidence", ascending=False)
                scores_df["Confidence"] = scores_df["Confidence"].apply(lambda x: f"{x:.1%}")
                st.dataframe(scores_df, use_container_width=True, hide_index=True)
            
            # Text metrics
            if show_metrics:
                st.markdown("### 📈 Document Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Word Count", len(text_input.split()), help="Total words in document")
                with col2:
                    st.metric("Character Count", len(text_input), help="Total characters including spaces")
                with col3:
                    read_time = max(1, len(text_input.split()) // 200)
                    st.metric("Reading Time", f"~{read_time} min", help="Estimated reading time")
                with col4:
                    st.metric("Avg Word Length", f"{sum(len(w) for w in text_input.split()) / len(text_input.split()):.1f}", help="Average characters per word")
        
        except Exception as e:
            st.error(
                f"**❌ Classification Failed.** An error occurred while analyzing your document. "
                f"Error details: {str(e)[:80]}"
            )


def named_entity_recognition(processor, show_metrics):
    """Named Entity Recognition interface - extract legal entities"""
    st.markdown("<h2 class='section-header'>🔍 Named Entity Recognition</h2>", unsafe_allow_html=True)
    
    st.markdown(
        "<div style='background: #f0f7ff; padding: 1em; border-radius: 8px; margin-bottom: 1.5em; border-left: 4px solid #00D9FF;'>"
        "<strong>💡 What this does:</strong> Identifies and extracts key entities from your legal text such as people, "
        "organizations, dates, clauses, and other important legal elements."
        "</div>",
        unsafe_allow_html=True
    )
    
    text_input = st.text_area(
        "Paste your legal document:",
        height=300,
        placeholder="Enter the legal document text to extract entities from (contracts, court filings, legal briefs, etc.)...",
        label_visibility="collapsed"
    )
    
    if st.button("🔎 Extract & Analyze Entities", key="ner", use_container_width=True):
        if not text_input.strip():
            st.error(
                "**⚠️ Please provide a document.** Paste or type legal text containing entities to extract. "
                "Example: 'John Smith, as represented by Attorney Jane Doe, entered into an agreement with Corp Inc.'"
            )
            return
        
        try:
            with st.spinner("🔍 Analyzing entities..."):
                entities = processor.extract_entities(text_input, group_by_type=True)
            
            st.markdown("<div class='success-message'>✅ Entity Extraction Complete</div>", unsafe_allow_html=True)
            
            if entities:
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_entities = sum(len(items) for items in entities.values())
                    st.metric("Total Entities Found", total_entities)
                with col2:
                    st.metric("Entity Types", len(entities))
                with col3:
                    st.metric("Accuracy", "High", help="Using fine-tuned legal BERT model")
                
                # Display by entity type with better presentation
                st.markdown("### 📌 Extracted Entities by Type")
                for entity_type, items in entities.items():
                    with st.expander(f"**{entity_type}** ({len(items)} found)", expanded=(len(items) <= 5)):
                        if items:
                            # Display entities in columns for better UX
                            cols = st.columns(2)
                            for idx, item in enumerate(items[:10]):
                                col = cols[idx % 2]
                                with col:
                                    st.markdown(
                                        f"<div class='entity-box'>• {item}</div>",
                                        unsafe_allow_html=True
                                    )
                            
                            if len(items) > 10:
                                st.info(f"**+{len(items) - 10} more** {entity_type.lower()} entities found. "
                                       "Showing top 10 for clarity.")
                
                # Statistical breakdown
                if show_metrics:
                    st.markdown("### 📊 Entity Distribution")
                    summary_data = {
                        "Entity Type": list(entities.keys()),
                        "Count": [len(items) for items in entities.values()]
                    }
                    import pandas as pd
                    summary_df = pd.DataFrame(summary_data).sort_values("Count", ascending=False)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.bar_chart(summary_df.set_index("Entity Type"))
                    with col2:
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "**ℹ️ No entities extracted.** The document may not contain standard legal entities "
                    "or the entities may be embedded in complex sentence structures. Try documents with clear names, dates, or organization references."
                )
        
        except Exception as e:
            st.error(
                f"**❌ Extraction Failed.** An error occurred during entity extraction. "
                f"Error: {str(e)[:80]}"
            )


def similarity_analysis(processor):
    """Similarity analysis interface - compare legal documents"""
    st.markdown("<h2 class='section-header'>🔗 Similarity Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown(
        "<div style='background: #f0f7ff; padding: 1em; border-radius: 8px; margin-bottom: 1.5em; border-left: 4px solid #00D9FF;'>"
        "<strong>💡 What this does:</strong> Compares legal documents or finds similar documents in your corpus. "
        "Useful for identifying duplicate clauses, related cases, or parallel contracts."
        "</div>",
        unsafe_allow_html=True
    )
    
    analysis_type = st.radio(
        "Choose analysis type:",
        ["Compare Two Documents", "Find Similar Documents"],
        horizontal=True
    )
    
    if analysis_type == "Compare Two Documents":
        col1, col2 = st.columns(2)
        
        with col1:
            text1 = st.text_area(
                "Document 1:",
                height=250,
                placeholder="First legal document...",
                label_visibility="visible"
            )
        
        with col2:
            text2 = st.text_area(
                "Document 2:",
                height=250,
                placeholder="Second legal document...",
                label_visibility="visible"
            )
        
        if st.button("📊 Calculate Similarity", key="compare", use_container_width=True):
            if not text1.strip() or not text2.strip():
                st.error("**⚠️ Both documents required.** Please enter text in both document fields.")
                return
            
            try:
                with st.spinner("📊 Calculating similarity..."):
                    similarity = processor.calculate_similarity(text1, text2)
                
                st.markdown("<div class='success-message'>✅ Analysis Complete</div>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Similarity Score", f"{similarity:.3f}", help="0 = completely different, 1 = identical")
                with col2:
                    percentage = similarity * 100
                    st.metric("Percentage Match", f"{percentage:.1f}%")
                with col3:
                    if similarity > 0.8:
                        match_type = "🟢 Very Similar"
                        interpretation = "These documents are highly similar - likely duplicates or near-identical"
                    elif similarity > 0.6:
                        match_type = "🟡 Moderately Similar"
                        interpretation = "These documents share significant content or structure"
                    elif similarity > 0.4:
                        match_type = "🟠 Somewhat Similar"
                        interpretation = "These documents have some overlap but differ substantially"
                    else:
                        match_type = "🔴 Dissimilar"
                        interpretation = "These documents are largely different in content and structure"
                    
                    st.metric("Classification", match_type)
                
                st.info(f"**Interpretation:** {interpretation}")
            
            except Exception as e:
                st.error(f"**❌ Similarity calculation failed:** {str(e)[:80]}")
    
    else:  # Find Similar Documents
        st.markdown("### 🔍 Search Query")
        query = st.text_area(
            "Enter the document you want to find matches for:",
            height=150,
            placeholder="Document to search for...",
            label_visibility="collapsed"
        )
        
        st.markdown("### 📚 Document Corpus")
        corpus_text = st.text_area(
            "Enter documents to search within (one document per line, separated by blank lines):",
            height=200,
            placeholder="Document 1\n\nDocument 2\n\nDocument 3...",
            label_visibility="collapsed"
        )
        
        if st.button("🔎 Find Similar Documents", key="find_similar", use_container_width=True):
            if not query.strip():
                st.error("**⚠️ Query document required.** Please enter the document you want to find matches for.")
                return
            
            if not corpus_text.strip():
                st.error("**⚠️ Corpus required.** Please enter documents to search within.")
                return
            
            # Parse corpus (separated by blank lines or newlines with filtering)
            corpus_docs = [doc.strip() for doc in corpus_text.split("\n\n") if doc.strip()]
            if len(corpus_docs) == 1:
                corpus_docs = [doc.strip() for doc in corpus_text.split("\n") if doc.strip()]
            
            try:
                with st.spinner(f"🔍 Searching through {len(corpus_docs)} documents..."):
                    results = processor.find_similar_documents(query, corpus_docs, top_k=min(5, len(corpus_docs)))
                
                st.markdown("<div class='success-message'>✅ Search Complete - Ranked Results Below</div>", unsafe_allow_html=True)
                
                if results:
                    for i, (doc, score) in enumerate(results, 1):
                        with st.expander(f"**#{i}** — Match: {score:.1%}", expanded=(i==1)):
                            st.write(doc)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Similarity", f"{score:.3f}")
                            with col2:
                                st.metric("Match Quality", 
                                         "Excellent" if score > 0.8 else ("Good" if score > 0.6 else "Fair"))
                else:
                    st.info("**ℹ️ No matches found.** Try adjusting your query or corpus content.")
            
            except Exception as e:
                st.error(f"**❌ Search failed:** {str(e)[:80]}")


def document_summarization(processor, show_metrics):
    """Document summarization interface - extract key information"""
    st.markdown("<h2 class='section-header'>📊 Document Summarization</h2>", unsafe_allow_html=True)
    
    st.markdown(
        "<div style='background: #f0f7ff; padding: 1em; border-radius: 8px; margin-bottom: 1.5em; border-left: 4px solid #00D9FF;'>"
        "<strong>💡 What this does:</strong> Automatically extracts key information from your legal document including "
        "document type, key entities, important clauses, and metadata. Perfect for quick document understanding."
        "</div>",
        unsafe_allow_html=True
    )
    
    text_input = st.text_area(
        "Paste your legal document:",
        height=300,
        placeholder="Enter legal document to summarize (longer documents work best for comprehensive analysis)...",
        label_visibility="collapsed"
    )
    
    if st.button("📋 Generate Intelligent Summary", key="summarize", use_container_width=True):
        if not text_input.strip():
            st.error(
                "**⚠️ Document required.** Please enter legal text to summarize. "
                "Works best with documents of 200+ words for comprehensive analysis."
            )
            return
        
        try:
            with st.spinner("🤖 Generating intelligent summary..."):
                summarizer = DocumentSummarizer(processor)
                summary = summarizer.get_document_summary(text_input)
            
            st.markdown("<div class='success-message'>✅ Summary Generated Successfully</div>", unsafe_allow_html=True)
            
            # Overview metrics
            st.markdown("### 📋 Document Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Document Type", summary["type"].upper(), help="Identified document classification")
            with col2:
                st.metric("Confidence", f"{summary['confidence']:.0%}", help="Classification confidence score")
            with col3:
                st.metric("Word Count", summary["word_count"], help="Total words in document")
            with col4:
                read_time = summary.get('estimated_reading_time_minutes', max(1, summary["word_count"] // 200))
                st.metric("Reading Time", f"~{read_time} min", help="Estimated time to read")
            
            # Key entities section
            if summary.get("key_entities"):
                st.markdown("### 🔑 Key Entities Identified")
                entity_cols = st.columns(2)
                col_idx = 0
                
                for entity_type, entities in summary["key_entities"].items():
                    if entities:
                        with entity_cols[col_idx % 2]:
                            st.markdown(
                                f"<div class='entity-box'><strong>{entity_type}</strong><br>"
                                f"{', '.join(entities[:3])}"
                                + (f"<br>+{len(entities)-3} more..." if len(entities) > 3 else "")
                                + "</div>",
                                unsafe_allow_html=True
                            )
                            col_idx += 1
            
            # Key clauses section
            if summary.get("key_clauses"):
                st.markdown("### 📌 Key Clauses & Provisions")
                for i, clause in enumerate(summary["key_clauses"][:5], 1):
                    with st.expander(f"Clause {i}", expanded=(i==1)):
                        st.write(clause[:200] + ("..." if len(clause) > 200 else ""))
                
                if len(summary.get("key_clauses", [])) > 5:
                    st.info(f"**+{len(summary['key_clauses'])-5} additional clauses** identified. Showing top 5.")
            
            # Document metadata
            if show_metrics:
                st.markdown("### 📊 Document Analysis Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Unique Entities", len(summary.get("key_entities", {})))
                with col2:
                    st.metric("Key Clauses Found", len(summary.get("key_clauses", [])))
        
        except Exception as e:
            st.error(
                f"**❌ Summarization Failed.** An error occurred during analysis. "
                f"Error: {str(e)[:80]}"
            )


def batch_processing(processor):
    """Batch processing interface - process multiple documents efficiently"""
    st.markdown("<h2 class='section-header'>⚡ Batch Processing</h2>", unsafe_allow_html=True)
    
    st.markdown(
        "<div style='background: #f0f7ff; padding: 1em; border-radius: 8px; margin-bottom: 1.5em; border-left: 4px solid #00D9FF;'>"
        "<strong>💡 What this does:</strong> Efficiently process large numbers of legal documents in batch. "
        "Upload a text file and process all documents with a single task (classification, NER, or summarization)."
        "</div>",
        unsafe_allow_html=True
    )
    
    # File upload section
    st.markdown("### 📄 Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload a text file with one document per line:",
        type=["txt"],
        help="Plain text file with legal documents separated by newlines"
    )
    
    if uploaded_file:
        try:
            content = uploaded_file.read().decode("utf-8")
            documents = [line.strip() for line in content.split("\n") if line.strip()]
            
            if not documents:
                st.error("**⚠️ File is empty.** Please upload a file with at least one document.")
                return
            
            st.success(f"✅ Loaded {len(documents)} documents")
            
            # Task selection
            st.markdown("### ⚙️ Processing Configuration")
            task = st.selectbox(
                "Select processing task:",
                ["📄 Classify All Documents", "🔍 Extract All Entities", "📊 Summarize All Documents"],
                help="Choose which analysis to perform on all documents"
            )
            
            batch_size = st.slider("Documents per batch (for performance)", 1, 50, 10)
            
            if st.button("⚡ Start Processing", key="batch", use_container_width=True):
                task_map = {
                    "📄 Classify All Documents": "classify",
                    "🔍 Extract All Entities": "extract_entities",
                    "📊 Summarize All Documents": "summarize"
                }
                
                try:
                    with st.spinner(f"🔄 Processing {len(documents)} documents (batch size: {batch_size})..."):
                        results = batch_process_documents(
                            documents, processor, task=task_map[task], batch_size=batch_size
                        )
                    
                    st.markdown(
                        f"<div class='success-message'>✅ Successfully processed {len(results)} documents!</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Results display
                    import pandas as pd
                    import json
                    
                    if task_map[task] == "classify":
                        st.markdown("### 📊 Classification Results")
                        df = pd.DataFrame([
                            {
                                "Document": d[:50] + "..." if len(d) > 50 else d,
                                "Type": r.get("label", "Unknown").upper(),
                                "Confidence": f"{r.get('confidence', 0):.1%}"
                            }
                            for d, r in zip(documents[:len(results)], results)
                        ])
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            most_common = max(set([r.get("label") for r in results]), 
                                            key=[r.get("label") for r in results].count)
                            st.metric("Most Common Type", most_common)
                        with col2:
                            avg_confidence = sum(r.get("confidence", 0) for r in results) / len(results)
                            st.metric("Average Confidence", f"{avg_confidence:.1%}")
                        with col3:
                            st.metric("Documents Processed", len(results))
                    
                    # Export results
                    st.markdown("### 📥 Export Results")
                    
                    # CSV export
                    csv_data = pd.DataFrame(results).to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv_data,
                        file_name=f"batch_results_{task_map[task]}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # JSON export
                    json_data = json.dumps(results, indent=2)
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=json_data,
                        file_name=f"batch_results_{task_map[task]}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                except Exception as e:
                    st.error(
                        f"**❌ Batch Processing Failed.** Error during processing: {str(e)[:100]}"
                    )
        
        except Exception as e:
            st.error(
                f"**⚠️ File Reading Error.** Could not read the uploaded file. "
                f"Ensure it's a valid UTF-8 text file. Error: {str(e)[:80]}"
            )


if __name__ == "__main__":
    try:
        import pandas as pd
        main()
    except Exception as e:
        error_msg = str(e)
        st.error(
            f"**❌ Application Error.** An unexpected error occurred. "
            f"Details: {error_msg[:100]}"
        )
        st.info(
            "**How to fix this:**\n"
            "1. Make sure all dependencies are installed: `pip install -r requirements.txt`\n"
            "2. Check that PyTorch is installed: `python -c \"import torch; print(torch.__version__)\"`\n"
            "3. Verify transformers library: `python -c \"import transformers; print(transformers.__version__)\"`\n"
            "4. If using CPU, the app may run slower - consider using a GPU for better performance"
        )
        
        # Log the full traceback for debugging
        import traceback
        import logging
        logging.error(f"Full error trace:\n{traceback.format_exc()}")
