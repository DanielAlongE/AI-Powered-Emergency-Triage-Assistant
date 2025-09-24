"""
Concrete implementations of triage agents.
"""
from .random_agent import RandomTriageAgent
from .rule_based_agent import RuleBasedTriageAgent
from .llm_agent import LLMTriageAgent
from .hybrid_agent import HybridTriageAgent

# New agent for RAG + OpenAI integration
try:
    from .handbook_rag_openai_agent import HandbookRagOpenAiAgent
    HANDBOOK_RAG_AVAILABLE = True
except ImportError:
    HANDBOOK_RAG_AVAILABLE = False

if HANDBOOK_RAG_AVAILABLE:
    __all__ = ['RandomTriageAgent', 'RuleBasedTriageAgent', 'LLMTriageAgent', 'HybridTriageAgent', 'HandbookRagOpenAiAgent']
else:
    __all__ = ['RandomTriageAgent', 'RuleBasedTriageAgent', 'LLMTriageAgent', 'HybridTriageAgent']