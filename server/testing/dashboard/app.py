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
def load_test_data():
    """Load test data with caching."""
    try:
        runner = TriageTestRunner(max_workers=1)  # Single worker for UI responsiveness
        stats = runner.load_test_data()
        return runner, stats
    except Exception as e:
        st.error(f"Failed to load test data: {e}")
        return None, None


@st.cache_data(ttl=60)  # Cache for 1 minute
def run_agent_test(agent_config: Dict[str, Any], limit: int = 50):
    """Run agent test with caching."""
    runner, _ = load_test_data()
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
        else:
            return None

        # Run test
        results = runner.test_agent(agent, limit=limit)
        summary = runner.generate_summary(results)

        return {
            'agent': agent,
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
        agent_type = st.selectbox(
            "Agent Type",
            options=['random', 'rule_based'] + (['llm', 'hybrid'] if LLM_AVAILABLE else []),
            format_func=lambda x: {
                'random': 'Random Agent',
                'rule_based': 'Rule-Based Agent',
                'llm': 'LLM Agent',
                'hybrid': 'Hybrid Agent'
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

        # Test parameters
        st.divider()
        st.subheader("🧪 Test Parameters")
        test_limit = st.slider("# of Test Cases to Run", 10, 150, 150, 10)

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
                test_result = run_agent_test(agent_config, test_limit)

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
                    'Time (ms)': f"{r.processing_time * 1000:.1f}"
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
                        'agent_name': r.agent_name
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