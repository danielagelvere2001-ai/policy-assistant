"""Company Policy Assistant: three ways to answer policy questions."""
from .policies import Answer, Policy, load_policies
from .rules import RulesSearch

__all__ = ["Answer", "Policy", "load_policies", "RulesSearch", "build_all"]


def build_all(policies=None, client=None):
    """Return the three approaches, keyed by short id."""
    from .gemini import GeminiClient
    from .llm import LLMFullContext, LLMVectorIndex

    policies = policies or load_policies()
    client = client or GeminiClient()
    return {
        "rules": RulesSearch(policies),
        "llm": LLMFullContext(policies, client),
        "vector": LLMVectorIndex(policies, client),
    }
