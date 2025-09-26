"""
Agent module for emergency triage assessment.
"""
from .base import BaseTriageAgent
from .implementations.random_agent import RandomTriageAgent
from .implementations.rule_based_agent import RuleBasedTriageAgent
from .implementations.llm_agent import LLMTriageAgent
from .implementations.hybrid_agent import HybridTriageAgent

# Try to import optional agents
try:
    from .implementations.handbook_rag_openai_agent import HandbookRagOpenAiAgent
    HANDBOOK_RAG_OPENAI_AVAILABLE = True
except ImportError:
    HANDBOOK_RAG_OPENAI_AVAILABLE = False

try:
    from .implementations.handbook_rag_ollama_agent import HandbookRagOllamaAgent
    HANDBOOK_RAG_OLLAMA_AVAILABLE = True
except ImportError:
    HANDBOOK_RAG_OLLAMA_AVAILABLE = False

__all__ = ['BaseTriageAgent', 'RandomTriageAgent', 'RuleBasedTriageAgent', 'LLMTriageAgent', 'HybridTriageAgent']

if HANDBOOK_RAG_OPENAI_AVAILABLE:
    __all__.append('HandbookRagOpenAiAgent')

if HANDBOOK_RAG_OLLAMA_AVAILABLE:
    __all__.append('HandbookRagOllamaAgent')