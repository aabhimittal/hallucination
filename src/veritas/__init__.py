"""VERITAS: Verification-Enhanced Retrieval with Iterative Truth Assessment
and Scoring — a hallucination-reduction RAG pipeline that works on top of any
LLM.
"""

from .chunking import Chunk, Document, chunk_corpus, chunk_document, documents_from_texts
from .llm import (
    AnthropicClient,
    HFInferenceClient,
    LLMClient,
    MockLLM,
    OpenAICompatClient,
)
from .pipeline import (
    BaselineRAG,
    BaselineResult,
    PipelineConfig,
    VeritasPipeline,
    VeritasResult,
)
from .prompts import ABSTAIN_TEXT, STAGE_TEMPERATURES
from .retrieval import HybridRetriever, ScoredChunk
from .verification import ClaimVerdict, Verdict, lexical_entailment

__version__ = "0.1.0"

__all__ = [
    "ABSTAIN_TEXT",
    "AnthropicClient",
    "BaselineRAG",
    "BaselineResult",
    "Chunk",
    "ClaimVerdict",
    "Document",
    "HFInferenceClient",
    "HybridRetriever",
    "LLMClient",
    "MockLLM",
    "OpenAICompatClient",
    "PipelineConfig",
    "STAGE_TEMPERATURES",
    "ScoredChunk",
    "Verdict",
    "VeritasPipeline",
    "VeritasResult",
    "chunk_corpus",
    "chunk_document",
    "documents_from_texts",
    "lexical_entailment",
]
