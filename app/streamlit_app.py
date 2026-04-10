"""
Streamlit UI for Legal BERT NLP Application
Interactive document processing interface
"""
import streamlit as st
import torch
import sys
import os
from pathlib import Path
import pandas as pd
import PyPDF2
from io import BytesIO
import docx

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from inference.processor import LegalDocumentProcessor, DocumentSummarizer, batch_process_documents
from preprocessing.text_cleaner import TextCleaner


# Page configuration with premium styling
st.set_page_config(
    page_title="⚖️ Legal BERT NLP - Professional Document Analysis",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Legal BERT NLP - Advanced NLP for Legal Documents"}
)

# Prevent rerun on page load
if 'last_tab' not in st.session_state:
    st.session_state.last_tab = None

# Premium design system with advanced UI and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    :root {
        --primary: #0F3460;
        --primary-light: #1a4d7a;
        --secondary: #533483;
        --accent: #00D9FF;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --neutral-50: #f9fafb;
        --neutral-100: #f3f4f6;
        --neutral-200: #e5e7eb;
        --neutral-700: #374151;
        --neutral-900: #111827;
    }
    
    * {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    code {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Enhanced Animation Library */
    @keyframes fadeInSlide {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }
    
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 5px rgba(0, 217, 255, 0.5);
        }
        50% {
            box-shadow: 0 0 20px rgba(0, 217, 255, 0.8);
        }
    }
    
    @keyframes bounce {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-8px);
        }
    }
    
    /* Main Header Styling */
    .main-header {
        font-size: 3.5em;
        font-weight: 700;
        background: linear-gradient(135deg, #0F3460 0%, #533483 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.25em;
        letter-spacing: -1.5px;
        animation: fadeInSlide 0.6s ease-out;
    }
    
    .subheader-text {
        text-align: center;
        font-size: 1.2em;
        color: #6b7280;
        margin-bottom: 2em;
        font-weight: 500;
        animation: fadeInSlide 0.7s ease-out 0.1s both;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2em;
        font-weight: 700;
        color: #0F3460;
        border-bottom: 3px solid #00D9FF;
        padding-bottom: 0.75em;
        margin: 2em 0 1.5em 0;
        display: inline-block;
        animation: slideInLeft 0.5s ease-out;
    }
    
    /* Metric and Info Boxes */
    .metric-box {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        padding: 2em;
        border-radius: 12px;
        margin: 1.2em 0;
        border: 1px solid #e5e7eb;
        transition: all 0.3s cubic-bezier(0.25, 1, 0.5, 1);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        animation: fadeInSlide 0.5s ease-out;
    }
    
    .metric-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(15, 52, 96, 0.12);
        border-color: #00D9FF;
    }
    
    .entity-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #ede9fe 100%);
        padding: 1.2em;
        border-left: 4px solid #00D9FF;
        margin: 0.8em 0;
        border-radius: 8px;
        transition: all 0.2s ease;
        animation: slideInLeft 0.4s ease-out;
    }
    
    .entity-box:hover {
        border-left-color: #533483;
        background: linear-gradient(135deg, #e0f2fe 0%, #ede9fe 100%);
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(83, 52, 131, 0.15);
    }
    
    /* Alert Messages */
    .success-message {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        padding: 1.2em;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        font-weight: 500;
        animation: fadeInSlide 0.4s ease-out;
    }
    
    .error-message {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #7f1d1d;
        padding: 1.2em;
        border-radius: 8px;
        border-left: 4px solid #ef4444;
        font-weight: 500;
        animation: fadeInSlide 0.4s ease-out;
    }
    
    .info-message {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #0c2d6b;
        padding: 1.2em;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        font-weight: 500;
        animation: fadeInSlide 0.4s ease-out;
    }
    
    .warning-message {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #78350f;
        padding: 1.2em;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        font-weight: 500;
        animation: fadeInSlide 0.4s ease-out;
    }
    
    /* Task Labels and Controls */
    .task-label {
        font-size: 1.1em;
        font-weight: 600;
        color: #0F3460;
        margin: 1.5em 0 0.75em 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.85em;
    }
    
    /* Upload Boxes */
    .upload-box {
        border: 2px dashed #00D9FF;
        border-radius: 12px;
        padding: 2em;
        text-align: center;
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.05) 0%, rgba(83, 52, 131, 0.05) 100%);
        transition: all 0.3s ease;
        animation: fadeInSlide 0.5s ease-out;
    }
    
    .upload-box:hover {
        border-color: #533483;
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(83, 52, 131, 0.1) 100%);
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
    }
    
    /* Buttons */
    .button-primary {
        background: linear-gradient(135deg, #0F3460 0%, #533483 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75em 2em !important;
        transition: all 0.3s ease !important;
        position: relative;
        overflow: hidden;
    }
    
    .button-primary:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }
    
    .button-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(15, 52, 96, 0.3) !important;
    }
    
    .button-primary:hover:before {
        left: 100%;
    }
    
    /* Confidence Bars */
    .confidence-bar {
        height: 8px;
        background: #e5e7eb;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5em 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981 0%, #00D9FF 100%);
        transition: width 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        animation: slideInLeft 0.6s ease-out;
    }
    
    /* Data Table Enhancements */
    [data-testid="dataframe"] {
        border-radius: 8px;
        overflow: hidden;
    }
    
    [data-testid="dataframe"] thead {
        background: linear-gradient(135deg, #0F3460 0%, #533483 100%);
        color: white;
    }
    
    [data-testid="dataframe"] tbody tr:hover {
        background-color: rgba(0, 217, 255, 0.1);
    }
    
    /* Expander Styling */
    [data-testid="expanderHeader"] {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
    }
    
    [data-testid="expanderHeader"]:hover {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        border-color: #00D9FF;
    }
    
    /* Tab Styling */
    [data-testid="stTabs"] button {
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    [data-testid="stTabs"] button:hover {
        color: #00D9FF;
    }
    
    /* Select Box Enhancements */
    [data-testid="selectbox"] select {
        border: 2px solid #e5e7eb !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="selectbox"] select:hover {
        border-color: #00D9FF !important;
    }
    
    [data-testid="selectbox"] select:focus {
        border-color: #0F3460 !important;
        box-shadow: 0 0 0 3px rgba(15, 52, 96, 0.1) !important;
    }
    
    /* Text Input Enhancements */
    [data-testid="textinput"] input,
    [data-testid="stTextArea"] textarea {
        border: 2px solid #e5e7eb !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stTextArea"] textarea {
        min-height: 120px !important;
    }
    
    [data-testid="textinput"] input:hover,
    [data-testid="stTextArea"] textarea:hover {
        border-color: #00D9FF !important;
    }
    
    [data-testid="textinput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: #0F3460 !important;
        box-shadow: 0 0 0 3px rgba(15, 52, 96, 0.1) !important;
    }
    
    /* Slider Enhancements */
    [data-testid="slider"] {
        margin: 2em 0;
    }
    
    /* Checkbox Enhancements */
    [data-testid="checkbox"] {
        transition: all 0.2s ease;
    }
    
    [data-testid="checkbox"]:hover {
        transform: scale(1.05);
    }
    
    /* Spinner Animation */
    [data-testid="stSpinner"] {
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f9fafb 0%, #f3f4f6 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    [data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {
        background-color: #e5e7eb;
        transform: translateX(2px);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5em;
        }
        
        .section-header {
            font-size: 1.5em;
        }
        
        .metric-box {
            padding: 1.5em;
        }
    }
</style>
""", unsafe_allow_html=True)

# ── Sample texts ───────────────────────────────────────────────────────────────
SAMPLE_TEXTS = {
    "Contract": (
        "This Service Agreement (the \"Agreement\") is entered into as of January 1, 2025, "
        "between ABC Corporation, a Delaware corporation (\"Company\"), and John Smith "
        "(\"Consultant\"). The Company hereby engages Consultant to provide software development "
        "services. Consultant shall be paid $150 per hour for services rendered. Either party may "
        "terminate this Agreement upon thirty (30) days written notice. This Agreement shall be "
        "governed by the laws of the State of California."
    ),
    "Court Case": (
        "IN THE UNITED STATES DISTRICT COURT FOR THE SOUTHERN DISTRICT OF NEW YORK. "
        "Plaintiff Jane Doe v. Defendant XYZ Corp., Case No. 2025-CV-01234. "
        "Judge Hon. Robert Williams presiding. Plaintiff alleges breach of contract and "
        "seeks compensatory damages of $500,000. Defendant moves to dismiss under Rule 12(b)(6) "
        "for failure to state a claim upon which relief can be granted."
    ),
    "Statute": (
        "Section 42. Prohibition of Unfair Trade Practices. (a) No person shall engage in "
        "unfair methods of competition or unfair or deceptive acts or practices in commerce. "
        "(b) Any violation of subsection (a) shall be subject to a civil penalty not to exceed "
        "$10,000 per violation. (c) The Attorney General is authorized to bring civil actions "
        "to enforce the provisions of this section."
    ),
    "Appeal": (
        "NOTICE OF APPEAL. Appellant Smith Corp. hereby appeals to the United States Court of "
        "Appeals for the Ninth Circuit from the final judgment entered on December 15, 2024 by "
        "the Honorable District Court Judge Adams in Case No. 2024-CV-5678. The basis for this "
        "appeal is that the lower court erred in granting summary judgment on the antitrust claims "
        "and misapplied the relevant legal standard under Sherman Act Section 1."
    ),
}


def extract_text_from_file(uploaded_file):
    """Extract text from various document types"""
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    try:
        if file_extension == '.pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        
        elif file_extension == '.docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        
        elif file_extension in ['.txt', '.md']:
            return uploaded_file.read().decode('utf-8')
        
        elif file_extension == '.csv':
            df = pd.read_csv(uploaded_file)
            return df.to_string()
        
        else:
            st.error(f"❌ Unsupported file type: {file_extension}")
            return None
    
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)[:100]}")
        return None


@st.cache_resource
def load_models():
    """Load models (cached for performance)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = LegalDocumentProcessor(device=device)
    return processor


def main():
    # Header
    st.markdown(
        "<h1 class='main-header'>⚖️ Legal BERT NLP Platform</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<div class='subheader-text'>"
        "Professional AI-Powered Legal Document Analysis"
        "</div>",
        unsafe_allow_html=True,
    )
    
    st.markdown(
        "<div style='text-align: center; color: #6b7280; font-size: 0.95em; margin-bottom: 2em;'>"
        "📄 Classification • 🔍 Entity Recognition • 🔗 Similarity • 📊 Summarization • ⚡ Batch Processing"
        "</div>",
        unsafe_allow_html=True
    )
    
    # Sidebar configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    
    task = st.sidebar.radio(
        "Select Your Task:",
        [
            "📄 Document Classification",
            "🔍 Named Entity Recognition",
            "🔗 Similarity Analysis",
            "📊 Document Summarization",
            "⚡ Batch Processing",
            "📤 Multi-File Upload"
        ],
        key="task_selector"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Display Options")
    show_metrics = st.sidebar.checkbox("Show detailed metrics", value=True, help="Display additional analysis metrics")
    show_probabilities = st.sidebar.checkbox("Show confidence scores", value=True, help="Display per-class confidence breakdown")
    show_raw_output = st.sidebar.checkbox("Show raw model output", value=False, help="Display raw model predictions")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 System Information")
    
    try:
        with st.spinner("Loading AI models…"):
            processor = load_models()
        
        st.sidebar.success("✅ Models loaded")
        st.sidebar.markdown(f"""
        **Model Configuration**
        - Base Model: {config.DEFAULT_MODEL}
        - Framework: PyTorch + Transformers
        - Hardware: {'🚀 GPU (CUDA)' if torch.cuda.is_available() else '💻 CPU'}
        - Document Types: {len(config.CLASSIFICATION_LABELS)}
        """)
    
    except Exception as e:
        st.sidebar.error(f"⚠️ Model loading failed")
        st.error(f"**Unable to load models:** {str(e)[:60]}")
        st.info("Please ensure all dependencies are installed: `pip install -r requirements.txt`")
        return

    # ── Task routing ──
    if task == "📄 Document Classification":
        document_classification(processor, show_metrics, show_probabilities, show_raw_output)
    
    elif task == "🔍 Named Entity Recognition":
        named_entity_recognition(processor, show_metrics)
    elif task == "🔗 Similarity Analysis":
        similarity_analysis(processor)
    elif task == "📊 Document Summarization":
        document_summarization(processor, show_metrics)
    elif task == "⚡ Batch Processing":
        batch_processing(processor)
    
    elif task == "📤 Multi-File Upload":
        multi_file_upload(processor, show_metrics)


def document_classification(processor, show_metrics, show_probabilities, show_raw_output=False):
    """Document classification interface - identify document type"""
    st.markdown("<div class='sec-title'>📄 Document Classification</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='callout'>"
        "<strong>What this does:</strong> Analyzes your legal document and classifies it into one of "
        f"{len(config.CLASSIFICATION_LABELS)} types (contract, case, appeal, statute) with confidence scoring."
        "</div>",
        unsafe_allow_html=True,
    )

    col_input, col_types = st.columns([3, 1])

    with col_input:
        # Sample text quick-fill
        sample_choice = st.selectbox(
            "💡 Load a sample document (optional):",
            ["— choose a sample —"] + list(SAMPLE_TEXTS.keys()),
            key="clf_sample",
        )
        prefill = SAMPLE_TEXTS.get(sample_choice, "")
        text_input = st.text_area(
            "Paste your legal document:",
            value=prefill,
            height=240,
            placeholder="Paste or type your legal document here…",
            label_visibility="collapsed",
            key="clf_text",
        )

    with col_types:
        st.markdown("**Supported types**")
        for label in config.CLASSIFICATION_LABELS:
            st.markdown(f"• `{label}`")

    # Empty state
    if not text_input.strip():
        st.markdown(
            "<div class='empty-state'>"
            "<div class='es-icon'>📄</div>"
            "<h3>No document provided yet</h3>"
            "<p>Paste a legal document above or choose a sample to try the classifier.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if st.button("🔍 Classify Document", key="classify", use_container_width=True):
        try:
            with st.spinner("Analyzing document…"):
                result = processor.classify_document(text_input, return_proba=show_probabilities)

            st.markdown("<div class='pill-success'>✅ Classification complete</div>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                confidence_pct = int(result["confidence"] * 100)
                st.markdown(
                    f"<div class='result-card'>"
                    f"<div class='result-label'>Document Type</div>"
                    f"<div class='result-value'>{result['label'].upper()}</div>"
                    f"<div class='result-sub'>↑ {result['confidence']:.1%} confidence</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence_pct}%",
                    delta="High" if confidence_pct >= 80 else ("Fair" if confidence_pct >= 60 else "Low"),
                )
            with col3:
                reliability = "Reliable" if confidence_pct >= 80 else ("Moderate" if confidence_pct >= 60 else "Review")
                st.metric("Reliability", reliability)

            if show_probabilities and "all_scores" in result:
                st.markdown("**Classification scores by type**")
                scores_df = __import__("pandas").DataFrame(
                    [(k, v) for k, v in result["all_scores"].items()],
                    columns=["Document Type", "Confidence"],
                ).sort_values("Confidence", ascending=False)
                scores_df["Confidence"] = scores_df["Confidence"].apply(lambda x: f"{x:.1%}")
                st.dataframe(scores_df, use_container_width=True, hide_index=True)

            if show_metrics:
                st.markdown("**Document statistics**")
                words = text_input.split()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Words", len(words))
                c2.metric("Characters", len(text_input))
                c3.metric("Reading time", f"~{max(1, len(words) // 200)} min")
                c4.metric("Avg word length", f"{sum(len(w) for w in words) / len(words):.1f}")

        except Exception as e:
            st.error(f"**Classification failed.** {str(e)[:120]}")


def named_entity_recognition(processor, show_metrics):
    """Named Entity Recognition interface - extract legal entities"""
    st.markdown("<div class='sec-title'>🔍 Named Entity Recognition</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='callout'>"
        "<strong>What this does:</strong> Identifies and extracts key entities from your legal text — "
        "people, judges, organizations, dates, and legal clauses — with per-entity confidence."
        "</div>",
        unsafe_allow_html=True,
    )

    sample_choice = st.selectbox(
        "💡 Load a sample document (optional):",
        ["— choose a sample —"] + list(SAMPLE_TEXTS.keys()),
        key="ner_sample",
    )
    prefill = SAMPLE_TEXTS.get(sample_choice, "")
    text_input = st.text_area(
        "Paste your legal document:",
        value=prefill,
        height=260,
        placeholder="Paste legal text here to extract named entities…",
        label_visibility="collapsed",
        key="ner_text",
    )

    if not text_input.strip():
        st.markdown(
            "<div class='empty-state'>"
            "<div class='es-icon'>🔍</div>"
            "<h3>No document provided yet</h3>"
            "<p>Paste legal text above or select a sample. "
            "Example: <em>'John Smith, as represented by Attorney Jane Doe, entered into an agreement with Corp Inc.'</em></p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if st.button("🔎 Extract Entities", key="ner", use_container_width=True):
        try:
            with st.spinner("Analyzing entities…"):
                entities = processor.extract_entities(text_input, group_by_type=True)

            st.markdown("<div class='pill-success'>✅ Entity extraction complete</div>", unsafe_allow_html=True)

            if entities:
                total_entities = sum(len(items) for items in entities.values())
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Entities", total_entities)
                c2.metric("Entity Types", len(entities))
                c3.metric("Model", "Legal BERT")

                st.markdown("**Extracted entities by type**")
                for entity_type, items in entities.items():
                    if not items:
                        continue
                    with st.expander(f"**{entity_type}** — {len(items)} found", expanded=(len(items) <= 5)):
                        chips_html = "".join(
                            f"<span class='entity-chip'>{item}</span>"
                            for item in items[:15]
                        )
                        if len(items) > 15:
                            chips_html += f"<span class='entity-chip' style='background:#f1f5f9;color:#64748b;'>+{len(items)-15} more</span>"
                        st.markdown(chips_html, unsafe_allow_html=True)

                if show_metrics:
                    st.markdown("**Entity distribution**")
                    import pandas as pd
                    summary_df = __import__("pandas").DataFrame(
                        {"Entity Type": list(entities.keys()), "Count": [len(v) for v in entities.values()]}
                    ).sort_values("Count", ascending=False)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.bar_chart(summary_df.set_index("Entity Type"))
                    with c2:
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "**No entities found.** The document may not contain standard legal entities, "
                    "or they may be embedded in complex structures. Try documents with clear names, dates, or organizations."
                )

        except Exception as e:
            st.error(f"**Entity extraction failed.** {str(e)[:120]}")


def similarity_analysis(processor):
    """Similarity analysis interface - compare legal documents"""
    st.markdown("<div class='sec-title'>🔗 Similarity Analysis</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='callout'>"
        "<strong>What this does:</strong> Compares legal documents or searches a corpus for similar matches. "
        "Useful for duplicate clause detection, related-case research, or parallel contract review."
        "</div>",
        unsafe_allow_html=True,
    )

    analysis_type = st.radio(
        "Mode:",
        ["Compare Two Documents", "Find Similar in Corpus"],
        horizontal=True,
    )

    if analysis_type == "Compare Two Documents":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Document A**")
            text1 = st.text_area(
                "Document A:",
                height=220,
                placeholder="Paste first legal document…",
                label_visibility="collapsed",
                key="sim_doc1",
            )
        with col2:
            st.markdown("**Document B**")
            text2 = st.text_area(
                "Document B:",
                height=220,
                placeholder="Paste second legal document…",
                label_visibility="collapsed",
                key="sim_doc2",
            )

        if not text1.strip() and not text2.strip():
            st.markdown(
                "<div class='empty-state'>"
                "<div class='es-icon'>🔗</div>"
                "<h3>Enter two documents to compare</h3>"
                "<p>Paste a document in each field above, then click Compare.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            return

        if st.button("📊 Compare Documents", key="compare", use_container_width=True):
            if not text1.strip() or not text2.strip():
                st.error("**Both documents required.** Please fill in both fields.")
                return
            try:
                with st.spinner("Calculating similarity…"):
                    similarity = processor.calculate_similarity(text1, text2)

                st.markdown("<div class='pill-success'>✅ Comparison complete</div>", unsafe_allow_html=True)

                pct = similarity * 100
                if similarity > 0.8:
                    match_label, interpretation = "🟢 Very Similar", "Highly similar — likely duplicates or near-identical documents."
                elif similarity > 0.6:
                    match_label, interpretation = "🟡 Moderately Similar", "Significant shared content or structure."
                elif similarity > 0.4:
                    match_label, interpretation = "🟠 Somewhat Similar", "Some overlap but substantially different."
                else:
                    match_label, interpretation = "🔴 Dissimilar", "Largely different in content and structure."

                c1, c2, c3 = st.columns(3)
                c1.metric("Similarity Score", f"{similarity:.3f}", help="0 = completely different · 1 = identical")
                c2.metric("Percentage Match", f"{pct:.1f}%")
                c3.metric("Classification", match_label)
                st.info(f"**Interpretation:** {interpretation}")

            except Exception as e:
                st.error(f"**Similarity calculation failed.** {str(e)[:120]}")

    else:  # Find Similar in Corpus
        st.markdown("**Search query document**")
        query = st.text_area(
            "Query:",
            height=130,
            placeholder="Paste the document you want to find matches for…",
            label_visibility="collapsed",
            key="sim_query",
        )
        st.markdown("**Corpus** — one document per paragraph (blank line between docs)")
        corpus_text = st.text_area(
            "Corpus:",
            height=180,
            placeholder="Document 1\n\nDocument 2\n\nDocument 3…",
            label_visibility="collapsed",
            key="sim_corpus",
        )

        if not query.strip() and not corpus_text.strip():
            st.markdown(
                "<div class='empty-state'>"
                "<div class='es-icon'>🔍</div>"
                "<h3>Enter a query and corpus</h3>"
                "<p>Paste your search document and the corpus to search within, then click Find Similar.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            return

        if st.button("🔎 Find Similar Documents", key="find_similar", use_container_width=True):
            if not query.strip():
                st.error("**Query document required.**")
                return
            if not corpus_text.strip():
                st.error("**Corpus required.**")
                return

            corpus_docs = [d.strip() for d in corpus_text.split("\n\n") if d.strip()]
            if len(corpus_docs) == 1:
                corpus_docs = [d.strip() for d in corpus_text.split("\n") if d.strip()]

            try:
                with st.spinner(f"Searching {len(corpus_docs)} documents…"):
                    results = processor.find_similar_documents(query, corpus_docs, top_k=min(5, len(corpus_docs)))

                st.markdown("<div class='pill-success'>✅ Search complete — ranked results below</div>", unsafe_allow_html=True)

                if results:
                    for i, (doc, score) in enumerate(results, 1):
                        quality = "Excellent" if score > 0.8 else ("Good" if score > 0.6 else "Fair")
                        with st.expander(f"**#{i}** — {score:.1%} match · {quality}", expanded=(i == 1)):
                            st.write(doc)
                            ca, cb = st.columns(2)
                            ca.metric("Similarity", f"{score:.3f}")
                            cb.metric("Match Quality", quality)
                else:
                    st.info("**No matches found.** Try adjusting your query or adding more documents to the corpus.")

            except Exception as e:
                st.error(f"**Search failed.** {str(e)[:120]}")


def document_summarization(processor, show_metrics):
    """Document summarization interface - extract key information"""
    st.markdown("<div class='sec-title'>📊 Document Summarization</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='callout'>"
        "<strong>What this does:</strong> Generates a structured summary of your legal document — "
        "document type, key entities, important clauses, and metadata. Works best with 200+ word documents."
        "</div>",
        unsafe_allow_html=True,
    )

    sample_choice = st.selectbox(
        "💡 Load a sample document (optional):",
        ["— choose a sample —"] + list(SAMPLE_TEXTS.keys()),
        key="sum_sample",
    )
    prefill = SAMPLE_TEXTS.get(sample_choice, "")
    text_input = st.text_area(
        "Paste your legal document:",
        value=prefill,
        height=260,
        placeholder="Paste legal document text here…",
        label_visibility="collapsed",
        key="sum_text",
    )

    if not text_input.strip():
        st.markdown(
            "<div class='empty-state'>"
            "<div class='es-icon'>📊</div>"
            "<h3>No document provided yet</h3>"
            "<p>Paste a legal document above or select a sample to generate an intelligent summary.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if st.button("📋 Generate Summary", key="summarize", use_container_width=True):
        try:
            with st.spinner("Generating intelligent summary…"):
                summarizer = DocumentSummarizer(processor)
                summary = summarizer.get_document_summary(text_input)

            st.markdown("<div class='pill-success'>✅ Summary generated</div>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Document Type", summary["type"].upper())
            c2.metric("Confidence", f"{summary['confidence']:.0%}")
            c3.metric("Word Count", summary["word_count"])
            read_time = summary.get("estimated_reading_time_minutes", max(1, summary["word_count"] // 200))
            c4.metric("Reading Time", f"~{read_time} min")

            if summary.get("key_entities"):
                st.markdown("**Key entities identified**")
                for entity_type, ents in summary["key_entities"].items():
                    if ents:
                        chips = "".join(f"<span class='entity-chip'>{e}</span>" for e in ents[:5])
                        extra = f" <span class='entity-chip' style='background:#f1f5f9;color:#64748b;'>+{len(ents)-5} more</span>" if len(ents) > 5 else ""
                        st.markdown(
                            f"<div class='entity-group'>"
                            f"<div class='entity-group-title'>{entity_type}</div>"
                            f"{chips}{extra}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            if summary.get("key_clauses"):
                st.markdown("**Key clauses & provisions**")
                for i, clause in enumerate(summary["key_clauses"][:5], 1):
                    with st.expander(f"Clause {i}", expanded=(i == 1)):
                        st.write(clause[:300] + ("…" if len(clause) > 300 else ""))
                if len(summary.get("key_clauses", [])) > 5:
                    st.caption(f"+{len(summary['key_clauses']) - 5} additional clauses identified. Showing top 5.")

            if show_metrics:
                st.markdown("**Analysis details**")
                ca, cb = st.columns(2)
                ca.metric("Unique Entity Types", len(summary.get("key_entities", {})))
                cb.metric("Key Clauses Found", len(summary.get("key_clauses", [])))

        except Exception as e:
            st.error(f"**Summarization failed.** {str(e)[:120]}")


def batch_processing(processor):
    """Batch processing interface - process multiple documents efficiently"""
    st.markdown("<div class='sec-title'>⚡ Batch Processing</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='callout'>"
        "<strong>What this does:</strong> Process large numbers of legal documents at once. "
        "Upload a plain-text file (one document per line) and run classification, NER, or summarization across all of them."
        "</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a .txt file — one document per line:",
        type=["txt"],
        help="Plain-text file with legal documents, one per line",
    )

    if not uploaded_file:
        st.markdown(
            "<div class='empty-state'>"
            "<div class='es-icon'>⚡</div>"
            "<h3>No file uploaded yet</h3>"
            "<p>Upload a plain-text file with one legal document per line to start batch processing.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    try:
        content = uploaded_file.read().decode("utf-8")
        documents = [line.strip() for line in content.split("\n") if line.strip()]

        if not documents:
            st.error("**File is empty.** Please upload a file with at least one document.")
            return

        st.success(f"✅ Loaded **{len(documents)}** documents")

        st.markdown("**Processing configuration**")
        task = st.selectbox(
            "Select task:",
            ["📄 Classify All Documents", "🔍 Extract All Entities", "📊 Summarize All Documents"],
        )
        batch_size = st.slider("Batch size (documents per batch)", 1, 50, 10)

        if st.button("⚡ Start Batch Processing", key="batch", use_container_width=True):
            task_map = {
                "📄 Classify All Documents": "classify",
                "🔍 Extract All Entities": "extract_entities",
                "📊 Summarize All Documents": "summarize",
            }

            try:
                with st.spinner(f"Processing {len(documents)} documents…"):
                    results = batch_process_documents(
                        documents, processor, task=task_map[task], batch_size=batch_size
                    )

                st.markdown(
                    f"<div class='pill-success'>✅ Processed {len(results)} documents successfully</div>",
                    unsafe_allow_html=True,
                )

                import pandas as pd
                import json

                if task_map[task] == "classify":
                    st.markdown("**Classification results**")
                    df = pd.DataFrame([
                        {
                            "Document": d[:60] + "…" if len(d) > 60 else d,
                            "Type": r.get("label", "Unknown").upper(),
                            "Confidence": f"{r.get('confidence', 0):.1%}",
                        }
                        for d, r in zip(documents[: len(results)], results)
                    ])
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    c1, c2, c3 = st.columns(3)
                    labels = [r.get("label") for r in results]
                    most_common = max(set(labels), key=labels.count)
                    avg_conf = sum(r.get("confidence", 0) for r in results) / len(results)
                    c1.metric("Most Common Type", most_common)
                    c2.metric("Avg Confidence", f"{avg_conf:.1%}")
                    c3.metric("Documents Processed", len(results))

                st.markdown("**Export results**")
                col_csv, col_json = st.columns(2)
                with col_csv:
                    csv_data = pd.DataFrame(results).to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        data=csv_data,
                        file_name=f"batch_{task_map[task]}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with col_json:
                    json_data = json.dumps(results, indent=2)
                    st.download_button(
                        "📥 Download JSON",
                        data=json_data,
                        file_name=f"batch_{task_map[task]}.json",
                        mime="application/json",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"**Batch processing failed.** {str(e)[:120]}")

    except Exception as e:
        st.error(f"**File reading error.** Ensure it is a valid UTF-8 text file. {str(e)[:120]}")


def multi_file_upload(processor, show_metrics):
    """Multi-format file upload and analysis interface"""
    st.markdown("<h2 class='section-header'>📤 Multi-File Upload & Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown(
        "<div style='background: #f0f7ff; padding: 1em; border-radius: 8px; margin-bottom: 1.5em; border-left: 4px solid #00D9FF;'>"
        "<strong>💡 Supported Formats:</strong> PDF, DOCX, TXT, CSV, Markdown (MD) - Upload multiple files and analyze them together!"
        "</div>",
        unsafe_allow_html=True
    )
    
    # Upload section
    uploaded_files = st.file_uploader(
        "Upload legal documents (PDF, DOCX, TXT, CSV, MD):",
        type=["pdf", "docx", "txt", "csv", "md"],
        accept_multiple_files=True,
        help="You can upload multiple files at once. They will be processed individually or together."
    )
    
    if uploaded_files:
        st.markdown(f"### 📊 Uploaded Files ({len(uploaded_files)})")
        
        # Display uploaded files with file info
        file_info = []
        extracted_texts = {}
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", len(uploaded_files))
        
        total_chars = 0
        for uploaded_file in uploaded_files:
            try:
                text = extract_text_from_file(uploaded_file)
                extracted_texts[uploaded_file.name] = text
                total_chars += len(text)
                
                file_info.append({
                    "Filename": uploaded_file.name,
                    "Size (KB)": f"{uploaded_file.size / 1024:.1f}",
                    "Type": Path(uploaded_file.name).suffix.upper()[1:],
                    "Characters": len(text)
                })
            except Exception as e:
                st.warning(f"⚠️ Could not extract text from {uploaded_file.name}: {str(e)[:80]}")
        
        if file_info:
            with col2:
                st.metric("Total Size (KB)", f"{sum(f['size'] for f in uploaded_files) / 1024:.1f}" if uploaded_files else "0")
            with col3:
                st.metric("Total Characters", total_chars)
            with col4:
                st.metric("File Types", len(set(Path(f.name).suffix for f in uploaded_files)))
            
            # Show file details table
            st.markdown("#### File Details")
            df_files = pd.DataFrame(file_info)
            st.dataframe(df_files, use_container_width=True, hide_index=True)
            
            # Analysis options
            st.markdown("### ⚙️ Analysis Options")
            col1, col2 = st.columns(2)
            
            with col1:
                analysis_type = st.selectbox(
                    "Select analysis type",
                    ["Classify Each Document", "Extract Entities from All", "Compare Similarity", "Summarize All"]
                )
            
            with col2:
                combine_documents = st.checkbox(
                    "Combine all documents", 
                    value=False,
                    help="Treat all files as one large document (useful for related legal documents)"
                )
            
            if st.button("🚀 Analyze Files", use_container_width=True):
                try:
                    if combine_documents:
                        # Combine all texts
                        combined_text = "\n\n--- NEW DOCUMENT ---\n\n".join(
                            [f"[{name}]\n{text}" for name, text in extracted_texts.items()]
                        )
                        
                        if analysis_type == "Classify Each Document":
                            with st.spinner("🔄 Classifying combined document..."):
                                result = processor.classify_document(combined_text)
                            
                            st.markdown("### 📋 Combined Classification Result")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Classification", result['label'].upper())
                            with col2:
                                st.metric("Confidence", f"{result['confidence']:.1%}")
                            with col3:
                                st.metric("Files Analyzed", len(extracted_texts))
                            
                            if show_metrics and 'all_scores' in result:
                                st.markdown("#### Confidence Breakdown")
                                scores_df = pd.DataFrame([
                                    {"Class": k.upper(), "Score": f"{v:.2%}"}
                                    for k, v in result['all_scores'].items()
                                ])
                                st.dataframe(scores_df, hide_index=True)
                        
                        elif analysis_type == "Extract Entities from All":
                            with st.spinner("🔄 Extracting entities..."):
                                entities = processor.extract_entities(combined_text)
                            
                            st.markdown("### 🔍 Extracted Entities")
                            if entities:
                                entity_groups = {}
                                for ent in entities:
                                    tag = ent.get("entity_type", "OTHER")
                                    if tag not in entity_groups:
                                        entity_groups[tag] = []
                                    entity_groups[tag].append(ent["text"])
                                
                                for tag, texts in entity_groups.items():
                                    with st.expander(f"**{tag}** ({len(texts)})"):
                                        st.write(", ".join(set(texts)))
                        
                        elif analysis_type == "Summarize All":
                            with st.spinner("🔄 Summarizing documents..."):
                                summary = processor.summarize_document(combined_text)
                            
                            st.markdown("### 📊 Combined Summary")
                            st.write(summary)
                    
                    else:
                        # Process each file separately
                        st.markdown("### 📊 Individual File Analysis")
                        
                        for file_name, text in extracted_texts.items():
                            with st.expander(f"📄 {file_name}", expanded=False):
                                if analysis_type == "Classify Each Document":
                                    with st.spinner(f"Classifying {file_name}..."):
                                        result = processor.classify_document(text)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Type", result['label'].upper())
                                    with col2:
                                        st.metric("Confidence", f"{result['confidence']:.1%}")
                                    
                                    if show_metrics and 'all_scores' in result:
                                        st.markdown("**Confidence Scores:**")
                                        for cls, score in result['all_scores'].items():
                                            st.write(f"- {cls.upper()}: {score:.2%}")
                                
                                elif analysis_type == "Extract Entities from All":
                                    with st.spinner(f"Extracting from {file_name}..."):
                                        entities = processor.extract_entities(text)
                                    
                                    if entities:
                                        entity_groups = {}
                                        for ent in entities:
                                            tag = ent.get("entity_type", "OTHER")
                                            if tag not in entity_groups:
                                                entity_groups[tag] = []
                                            entity_groups[tag].append(ent["text"])
                                        
                                        for tag, texts in entity_groups.items():
                                            st.write(f"**{tag}:** {', '.join(set(texts))}")
                                
                                elif analysis_type == "Summarize All":
                                    with st.spinner(f"Summarizing {file_name}..."):
                                        summary = processor.summarize_document(text[:2000])
                                    st.write(summary)
                    
                    st.success("✅ Analysis Complete!")
                
                except Exception as e:
                    st.error(f"**❌ Analysis Failed:** {str(e)[:100]}")


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
