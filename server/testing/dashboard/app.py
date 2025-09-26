"""
Streamlit dashboard for emergency triage agent evaluation and monitoring.
"""
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add paths for imports
server_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(server_root / "src"))
sys.path.insert(0, str(server_root / "testing"))

# Import testing framework
from runner import TriageTestRunner
from utils.metrics import MetricsCalculator
from utils.visualization import VisualizationHelper

# Import agents
from agents.implementations.random_agent import RandomTriageAgent
from agents.implementations.rule_based_agent import RuleBasedTriageAgent

# LLM in Progress, not ready to use yet.
# try:
#     from agents.implementations.llm_agent import LLMTriageAgent
#     from agents.implementations.hybrid_agent import HybridTriageAgent
#     LLM_AVAILABLE = True
# except ImportError:
LLM_AVAILABLE = False

# Import handbook RAG agents
try:
    from agents.implementations.handbook_rag_openai_agent import HandbookRagOpenAiAgent
    HANDBOOK_RAG_OPENAI_AVAILABLE = True
except ImportError:
    HANDBOOK_RAG_OPENAI_AVAILABLE = False

try:
    from agents.implementations.handbook_rag_ollama_agent import HandbookRagOllamaAgent
    HANDBOOK_RAG_OLLAMA_AVAILABLE = True
except ImportError:
    HANDBOOK_RAG_OLLAMA_AVAILABLE = False


# Configure Streamlit page
st.set_page_config(
    page_title="Emergency Triage Agent Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2E86AB;
    text-align: center;
    margin-bottom: 1rem;
    border-bottom: 3px solid #A23B72;
    padding-bottom: 0.5rem;
}

.metric-card {
    background-color: #F18F01;
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    margin: 0.5rem 0;
}

.agent-status {
    padding: 0.5rem;
    border-radius: 0.3rem;
    margin: 0.2rem 0;
    font-weight: bold;
}

.status-good { background-color: #28a745; color: white; }
.status-warning { background-color: #ffc107; color: black; }
.status-error { background-color: #dc3545; color: white; }

.stDataFrame {
    border: 2px solid #2E86AB;
    border-radius: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_test_data(max_workers: int = 1):
    """Load test data with caching.

    Args:
        max_workers: Number of parallel workers for the test runner.
    """
    try:
        runner = TriageTestRunner(max_workers=max_workers)
        stats = runner.load_test_data()
        return runner, stats
    except Exception as e:
        st.error(f"Failed to load test data: {e}")
        return None, None


@st.cache_data(ttl=60)  # Cache for 1 minute
def run_agent_test(agent_config: Dict[str, Any], limit: int = 50, max_workers: int = 1, use_source_case_text: bool = False):
    """Run agent test with caching.

    Args:
        agent_config: Agent configuration dict.
        limit: Max number of test cases to run.
        max_workers: Number of parallel workers to use during testing.
        use_source_case_text: If True, use source case text instead of conversation.
    """
    runner, _ = load_test_data(max_workers)
    if runner is None:
        return None

    try:
        # Create agent based on config
        agent_type = agent_config['type']
        agent_name = agent_config['name']

        if agent_type == 'random':
            agent = RandomTriageAgent(agent_name, agent_config.get('params', {}))
        elif agent_type == 'rule_based':
            agent = RuleBasedTriageAgent(agent_name, agent_config.get('params', {}))
        elif agent_type == 'llm' and LLM_AVAILABLE:
            agent = LLMTriageAgent(agent_name, agent_config.get('params', {}))
        elif agent_type == 'hybrid' and LLM_AVAILABLE:
            agent = HybridTriageAgent(agent_name, agent_config.get('params', {}))
        elif agent_type == 'handbook_rag_openai' and HANDBOOK_RAG_OPENAI_AVAILABLE:
            # Set up environment for HandbookRagOpenAiAgent
            params = agent_config.get('params', {})
            if 'openai_api_key' in params:
                import os
                os.environ['OPENAI_API_KEY'] = params['openai_api_key']
            agent = HandbookRagOpenAiAgent(agent_name, params)
        elif agent_type == 'handbook_rag_ollama' and HANDBOOK_RAG_OLLAMA_AVAILABLE:
            # Create HandbookRagOllamaAgent with Modal support
            params = agent_config.get('params', {})

            # Set environment variables for Modal if provided
            if 'modal_endpoint' in params:
                import os
                os.environ['MODAL_ENDPOINT'] = params['modal_endpoint']

            agent = HandbookRagOllamaAgent(agent_name, params)
        else:
            return None

        # Run test
        results = runner.test_agent(agent, limit=limit, use_source_case_text=use_source_case_text)
        summary = runner.generate_summary(results)

        return {
            'agent_name': agent.name,
            'agent_type': agent_type,
            'results': results,
            'summary': summary,
            'timestamp': datetime.now()
        }
    except Exception as e:
        st.error(f"Test failed for {agent_config['name']}: {e}")
        return None


def create_metrics_cards(summary):
    """Create metric cards for summary display."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{summary.accuracy:.2%}",
            help="Overall prediction accuracy"
        )

    with col2:
        st.metric(
            "Total Cases",
            f"{summary.total_cases:,}",
            help="Number of test cases evaluated"
        )

    with col3:
        processing_time_ms = summary.avg_processing_time * 1000
        st.metric(
            "Avg Response Time",
            f"{processing_time_ms:.1f}ms",
            help="Average processing time per case"
        )

    with col4:
        if summary.avg_confidence:
            st.metric(
                "Avg Confidence",
                f"{summary.avg_confidence:.2%}",
                help="Average confidence in predictions"
            )
        else:
            st.metric("Avg Confidence", "N/A", help="Confidence not available")


def create_confusion_matrix(summary):
    """Create interactive confusion matrix."""
    cm = summary.confusion_matrix

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5'],
        y=['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5'],
        colorscale='Blues',
        hoverongaps=False,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        showscale=True
    ))

    fig.update_layout(
        title=f"Confusion Matrix - {summary.agent_name}",
        xaxis_title="Predicted ESI Level",
        yaxis_title="True ESI Level",
        height=500,
        width=600
    )

    return fig


def create_esi_performance_chart(summary):
    """Create ESI level performance chart."""
    esi_levels = []
    f1_scores = []
    precisions = []
    recalls = []
    support_counts = []

    for esi_level, metrics in summary.results_by_esi.items():
        if metrics['support'] > 0:  # Only include levels with test cases
            esi_levels.append(f"ESI {esi_level}")
            f1_scores.append(metrics['f1_score'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            support_counts.append(metrics['support'])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Precision',
        x=esi_levels,
        y=precisions,
        marker_color='lightcoral'
    ))

    fig.add_trace(go.Bar(
        name='Recall',
        x=esi_levels,
        y=recalls,
        marker_color='lightblue'
    ))

    fig.add_trace(go.Bar(
        name='F1 Score',
        x=esi_levels,
        y=f1_scores,
        marker_color='lightgreen'
    ))

    fig.update_layout(
        title=f"Performance by ESI Level - {summary.agent_name}",
        xaxis_title="ESI Level",
        yaxis_title="Score",
        barmode='group',
        height=400,
        yaxis=dict(range=[0, 1])
    )

    return fig


def main():
    """Main dashboard function."""

    # Header
    st.markdown('<h1 class="main-header">Emergency Triage Agent Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Test data info
        runner, stats = load_test_data()
        if stats:
            st.success(f"✅ Test data loaded: {stats['cases_with_esi']} cases")

            with st.expander("📊 Data Distribution"):
                esi_dist = stats['esi_distribution']
                for esi_level, count in esi_dist.items():
                    if count > 0:
                        st.write(f"ESI {esi_level}: {count} cases")
        else:
            st.error("❌ Failed to load test data")
            return

        st.divider()

        # Agent selection
        st.subheader("🤖 Select Agent")
        agent_options = ['random', 'rule_based']
        if LLM_AVAILABLE:
            agent_options.extend(['llm', 'hybrid'])
        if HANDBOOK_RAG_OPENAI_AVAILABLE:
            agent_options.append('handbook_rag_openai')
        if HANDBOOK_RAG_OLLAMA_AVAILABLE:
            agent_options.append('handbook_rag_ollama')

        agent_type = st.selectbox(
            "Agent Type",
            options=agent_options,
            format_func=lambda x: {
                'random': 'Random Agent',
                'rule_based': 'Rule-Based Agent',
                'llm': 'LLM Agent',
                'hybrid': 'Hybrid Agent',
                'handbook_rag_openai': 'Handbook RAG + OpenAI Agent',
                'handbook_rag_ollama': 'Handbook RAG + Ollama Agent'
            }.get(x, x)
        )

        agent_name = st.text_input("Agent Name", value=f"{agent_type.replace('_', ' ').title()} Agent")

        # Agent-specific parameters
        params = {}
        if agent_type == 'random':
            st.subheader("Random Agent Parameters")
            use_seed = st.checkbox("Use Random Seed", value=True)
            if use_seed:
                params['seed'] = st.number_input("Seed", value=42, min_value=0)

            use_weights = st.checkbox("Custom ESI Weights")
            if use_weights:
                st.write("ESI Level Weights:")
                w1 = st.slider("ESI 1", 0.0, 1.0, 0.2, 0.01)
                w2 = st.slider("ESI 2", 0.0, 1.0, 0.2, 0.01)
                w3 = st.slider("ESI 3", 0.0, 1.0, 0.2, 0.01)
                w4 = st.slider("ESI 4", 0.0, 1.0, 0.2, 0.01)
                w5 = st.slider("ESI 5", 0.0, 1.0, 0.2, 0.01)
                params['weights'] = [w1, w2, w3, w4, w5]

        elif agent_type == 'llm' and LLM_AVAILABLE:
            st.subheader("LLM Agent Parameters")
            params['model'] = st.selectbox("Model", ['llama3.2', 'llama3.1', 'llama2'])
            params['temperature'] = st.slider("Temperature", 0.0, 1.0, 0.3, 0.01)

        elif agent_type == 'hybrid' and LLM_AVAILABLE:
            st.subheader("Hybrid Agent Parameters")
            params['rule_weight'] = st.slider("Rule Weight", 0.0, 1.0, 0.3, 0.01)
            params['llm_weight'] = st.slider("LLM Weight", 0.0, 1.0, 0.7, 0.01)

        elif agent_type == 'handbook_rag_openai' and HANDBOOK_RAG_OPENAI_AVAILABLE:
            st.subheader("Handbook RAG + OpenAI Parameters")

            # OpenAI Configuration
            st.write("**OpenAI Configuration:**")
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Enter your OpenAI API key for this session"
            )
            if openai_key:
                params['openai_api_key'] = openai_key

            params['model'] = st.selectbox(
                "OpenAI Model",
                ['gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4.1-mini', 'gpt-5-mini', 'gpt-5'],
                index=0
            )
            params['temperature'] = st.slider("Temperature", 0.0, 1.0, 0.1, 0.01)
            params['max_questions'] = st.slider("Max Follow-up Questions", 1, 5, 3, 1)

            # Status indicators
            if openai_key:
                st.success("✅ OpenAI API Key provided")
            else:
                st.warning("⚠️ OpenAI API Key required for this agent")

            st.info("💡 This agent uses RAG (retrieval-augmented generation) with ESI protocol documents and red-flag detection for enhanced triage assessment.")

        elif agent_type == 'handbook_rag_ollama' and HANDBOOK_RAG_OLLAMA_AVAILABLE:
            st.subheader("Handbook RAG + Ollama Parameters")

            # Inference Configuration
            st.write("**Inference Configuration:**")

            inference_mode = st.selectbox(
                "Inference Location",
                options=["auto", "local", "modal"],
                index=0,
                format_func=lambda x: {
                    "auto": "🔄 Auto (Try Modal, fallback to Local)",
                    "local": "🖥️ Local CPU/GPU",
                    "modal": "☁️ Modal (Remote GPU)"
                }.get(x, x),
                help="Choose where to run inference. Auto tries Modal first and falls back to local if needed."
            )
            params['inference_mode'] = inference_mode

            # Show relevant configuration based on mode
            if inference_mode in ["local", "auto"]:
                st.write("**Local Ollama Settings:**")
                ollama_host = st.text_input(
                    "Ollama Host",
                    value="http://localhost:11434",
                    placeholder="http://localhost:11434",
                    help="Local Ollama server host URL"
                )
                if ollama_host:
                    params['ollama_host'] = ollama_host

            if inference_mode in ["modal", "auto"]:
                st.write("**Modal Settings:**")
                modal_endpoint = st.text_input(
                    "Modal Endpoint",
                    value="",
                    placeholder="https://your-app--ollama-serve.modal.run",
                    help="Your deployed Modal endpoint URL. Leave empty to use environment variable."
                )
                if modal_endpoint:
                    params['modal_endpoint'] = modal_endpoint

                modal_gpu = st.selectbox(
                    "GPU Type",
                    options=["t4", "a10g", "a100"],
                    index=0,
                    format_func=lambda x: {
                        "t4": "T4 (~$0.59/hr, good balance)",
                        "a10g": "A10G (~$1.32/hr, faster)",
                        "a100": "A100 (~$4/hr, fastest)"
                    }.get(x, x),
                    help="GPU type for Modal inference"
                )
                params['modal_gpu_type'] = modal_gpu

            params['model'] = st.selectbox(
                "Ollama Model",
                ['qwen2.5:0.5b', 'qwen2.5:1.5b', 'llama3.2:1b', 'gemma2:2b', 'phi3.5', 'llama3.2', 'llama3.1', 'llama2', 'qwen2:7b-instruct', 'gpt-oss:20b'],
                index=0,
                help="Select the Ollama model to use. Models ordered by speed (fastest first). Make sure the model is installed locally."
            )
            params['temperature'] = st.slider("Temperature", 0.0, 1.0, 0.3, 0.01)
            params['max_questions'] = st.slider("Max Follow-up Questions", 1, 5, 3, 1)

            # Status indicators based on inference mode
            if inference_mode == "local":
                st.info("🖥️ This agent uses local Ollama models for inference. Ensure Ollama is running and the model is installed.")
            elif inference_mode == "modal":
                st.info("☁️ This agent uses Modal for remote GPU inference. Ensure your Modal endpoint is deployed and accessible.")
            else:  # auto
                st.info("🔄 This agent tries Modal first, then falls back to local Ollama if needed.")

            # Connection checks
            local_connected = False
            modal_connected = False

            # Check local connection if applicable
            if inference_mode in ["local", "auto"] and 'ollama_host' in locals():
                try:
                    import ollama
                    ollama_client = ollama.Client(host=ollama_host)
                    models_response = ollama_client.list()

                    # Extract model names from Ollama ListResponse
                    available_models = []
                    if hasattr(models_response, 'models'):
                        for model_info in models_response.models:
                            if hasattr(model_info, 'model'):
                                available_models.append(model_info.model)

                    if available_models:
                        st.success(f"✅ Local Ollama connected. Models: {', '.join(available_models[:3])}{'...' if len(available_models) > 3 else ''}")
                        local_connected = True

                        # Check model availability
                        model_found = params['model'] in available_models
                        if not model_found:
                            # Fuzzy matching
                            model_base = params['model'].split(':')[0]
                            for available_model in available_models:
                                if available_model.split(':')[0] == model_base:
                                    model_found = True
                                    break

                        if model_found:
                            st.success(f"✅ Model '{params['model']}' available locally")
                        else:
                            st.warning(f"⚠️ Model '{params['model']}' not found locally")
                    else:
                        st.warning("⚠️ Local Ollama has no models")

                except Exception as e:
                    st.error(f"❌ Local Ollama connection failed: {str(e)}")

            # Check Modal connection if applicable
            if inference_mode in ["modal", "auto"] and modal_endpoint:
                try:
                    import httpx
                    import asyncio

                    async def check_modal():
                        async with httpx.AsyncClient(timeout=10) as client:
                            response = await client.get(f"{modal_endpoint}/health")
                            return response.json()

                    # Run async check
                    try:
                        result = asyncio.run(check_modal())
                        st.success(f"✅ Modal endpoint connected: {result.get('status', 'healthy')}")
                        modal_connected = True
                    except Exception:
                        st.error("❌ Modal endpoint not accessible")

                except Exception as e:
                    st.error(f"❌ Modal connection check failed: {str(e)}")

            elif inference_mode in ["modal", "auto"] and not modal_endpoint:
                st.warning("⚠️ Modal endpoint URL required for Modal inference")

            # Overall status
            if inference_mode == "local" and local_connected:
                st.success("🟢 Ready for local inference")
            elif inference_mode == "modal" and modal_connected:
                st.success("🟢 Ready for Modal inference")
            elif inference_mode == "auto" and (local_connected or modal_connected):
                ready_modes = []
                if modal_connected: ready_modes.append("Modal")
                if local_connected: ready_modes.append("Local")
                st.success(f"🟢 Ready for inference ({' + '.join(ready_modes)} available)")
            else:
                st.error("🔴 No inference backends available")

            st.info("💡 This agent uses RAG with ESI protocol documents and red-flag detection for enhanced triage assessment.")

        # Test parameters
        st.divider()
        st.subheader("🧪 Test Parameters")
        parallel_workers = st.number_input(
            "Parallel Workers",
            min_value=1,
            max_value=32,
            value=1,
            step=1,
            help="Number of threads to use when running test cases in parallel. Set to 1 to run sequentially."
        )
        test_limit = st.slider("Test Cases to Run", 10, 150, 150, 10)
        use_source_case_text = st.checkbox(
            "Use Source Case Text",
            value=False,
            help="Use the original source case text instead of the conversation. This provides a single turn with empty speaker and the meta.source_case_text as the message."
        )

        # Run test button
        run_test = st.button("Run Test", type="primary", use_container_width=True)

    # Main content area
    if run_test or 'last_test_result' in st.session_state:

        if run_test:
            # Configure agent
            agent_config = {
                'type': agent_type,
                'name': agent_name,
                'params': params
            }

            with st.spinner(f"Running test for {agent_name}..."):
                # Run test
                test_result = run_agent_test(agent_config, test_limit, max_workers=parallel_workers, use_source_case_text=use_source_case_text)

                if test_result:
                    st.session_state['last_test_result'] = test_result
                    st.success(f"✅ Test completed for {agent_name}")
                else:
                    st.error("❌ Test failed")
                    return

        # Display results
        if 'last_test_result' in st.session_state:
            result = st.session_state['last_test_result']
            summary = result['summary']

            # Test info
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"📊 Results: {summary.agent_name}")
            with col2:
                st.caption(f"Last updated: {result['timestamp'].strftime('%H:%M:%S')}")

            # Metrics cards
            create_metrics_cards(summary)

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                st.plotly_chart(
                    create_confusion_matrix(summary),
                    use_container_width=True
                )

            with col2:
                st.plotly_chart(
                    create_esi_performance_chart(summary),
                    use_container_width=True
                )

            # Detailed results table
            st.subheader("📋 Detailed Results")

            # Convert results to DataFrame
            results_data = []
            for r in result['results'][:20]:  # Show first 20 results
                results_data.append({
                    'Case ID': r.case_id,
                    'Expected ESI': r.expected_esi,
                    'Predicted ESI': r.predicted_esi,
                    'Correct': '✅' if r.correct else '❌',
                    'Confidence': f"{r.confidence:.2%}" if r.confidence else "N/A",
                    'Time (ms)': f"{r.processing_time * 1000:.1f}",
                    'Rationale': r.rationale[:100] + "..." if r.rationale and len(r.rationale) > 100 else (r.rationale or "N/A")
                })

            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)

            if len(result['results']) > 20:
                st.caption(f"Showing first 20 of {len(result['results'])} results")

            # Export options
            st.subheader("💾 Export Results")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📊 Download Full Results CSV"):
                    full_df = pd.DataFrame([{
                        'case_id': r.case_id,
                        'expected_esi': r.expected_esi,
                        'predicted_esi': r.predicted_esi,
                        'correct': r.correct,
                        'confidence': r.confidence,
                        'processing_time': r.processing_time,
                        'agent_name': r.agent_name,
                        'rationale': r.rationale
                    } for r in result['results']])

                    csv = full_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{summary.agent_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

    else:
        # Welcome message
        st.markdown("""
        ## Emergency Triage Agent Dashboard

        This dashboard allows you to:
        - **Test different triage agents** against the ESI test cases
        - **Compare performance** across multiple agents
        - **Visualize results** with charts
        - **Export results** for further analysis

        ### Getting Started
        1. Configure an agent in the sidebar
        2. Set test parameters
        3. Click "Run Test" to evaluate performance
        4. Explore the interactive results

        ### Available Agents
        - **Random Agent**: Baseline for comparison
        - **Rule-Based Agent**: Basic keyword and pattern matching
        """)

        if HANDBOOK_RAG_OPENAI_AVAILABLE:
            st.markdown("""
        - **Handbook RAG + OpenAI Agent**: Advanced AI-powered assessment using OpenAI with RAG and red-flag detection
            """)

        if HANDBOOK_RAG_OLLAMA_AVAILABLE:
            st.markdown("""
        - **Handbook RAG + Ollama Agent**: Advanced AI-powered assessment using local Ollama models with RAG and red-flag detection
            """)

        if LLM_AVAILABLE:
            st.markdown("""
        - **LLM Agent**: AI-powered assessment using Ollama
        - **Hybrid Agent**: Combines rule-based and LLM approaches
            """)
        else:
            st.warning("LLM agents not available")

        st.markdown("---")
        st.info("**Tip**: Start with the Random Agent to establish a baseline, then try the Rule-Based Agent for comparison.")


if __name__ == "__main__":
    main()
