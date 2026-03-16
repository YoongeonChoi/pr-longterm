from __future__ import annotations

from dataclasses import dataclass, field

from agent.context_manager import ContextManager
from agent.llm import HeuristicLLM, LLMProtocol
from compression.long_context import LongContextCompressor
from compression.strategies import ContextCompressor
from context.context_builder import ContextBuilder
from context.context_ranker import ContextRanker
from context.token_budget import TokenBudget
from memory.archival_memory import ArchivalMemory
from memory.episodic_memory import EpisodicMemory
from memory.hierarchy import HierarchicalMemoryRuntime
from memory.ingestion import DocumentIngestor, IngestionResult
from memory.manager import MemoryManager
from memory.memory_consolidation import MemoryConsolidation
from memory.semantic_memory import SemanticMemory
from memory.store import MemoryRecord, MemoryStore
from memory.working_memory import WorkingMemory
from retrieval.embedding import SimpleEmbeddingModel
from retrieval.hybrid import HybridRetriever
from retrieval.query_rewrite import QueryRewriter


@dataclass(slots=True)
class ChatResult:
    session_id: str
    response: str
    context_tokens: int
    was_compressed: bool
    observations: list[str] = field(default_factory=list)


class CognitiveMemoryAgent:
    def __init__(
        self,
        db_path: str = "memory_engine.db",
        llm: LLMProtocol | None = None,
        query_rewriter: QueryRewriter | None = None,
        token_budget: int = 1500,
    ) -> None:
        self.store = MemoryStore(db_path)
        self.memory_manager = MemoryManager(self.store)
        self.embedding_model = SimpleEmbeddingModel(dimensions=64)
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.retriever = HybridRetriever(
            store=self.store,
            embedding_model=self.embedding_model,
            query_rewriter=self.query_rewriter,
            graph_boost=0.2,
        )
        self.context_manager = ContextManager(
            token_budget=token_budget,
            retriever=self.retriever,
            compressor=ContextCompressor(),
        )
        self.context_builder = ContextBuilder(
            retriever=self.retriever,
            ranker=ContextRanker(),
            compressor=ContextCompressor(),
        )
        self.token_budget = TokenBudget(max_tokens=token_budget, reserved_response_tokens=300, reserved_system_tokens=80)

        # Explicit memory-layer components for portfolio readability.
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory(self.store)
        self.semantic_memory = SemanticMemory(store=self.store, embedding_model=self.embedding_model)
        self.archival_memory = ArchivalMemory(self.store)
        self.consolidation = MemoryConsolidation(store=self.store, manager=self.memory_manager)

        self.hierarchy = HierarchicalMemoryRuntime(memory_manager=self.memory_manager, retriever=self.retriever)
        self.ingestor = DocumentIngestor(store=self.store, embedding_model=self.embedding_model)
        self.long_context = LongContextCompressor()
        self.llm = llm or HeuristicLLM()
        self._conversation_by_session: dict[str, list[str]] = {}

    def ingest_document(
        self,
        session_id: str,
        title: str,
        text: str,
        chunk_size_words: int = 240,
        overlap_words: int = 40,
    ) -> IngestionResult:
        metadata = {"session_id": session_id}
        return self.ingestor.ingest_document(
            title=title,
            text=text,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            base_metadata=metadata,
        )

    def chat(self, session_id: str, user_input: str) -> ChatResult:
        history = self._conversation_by_session.setdefault(session_id, [])
        self.working_memory.set(session_id, "active_query", user_input)
        built = self.context_builder.build(
            query=user_input,
            conversation=history[-20:],
            retrieval_limit=12,
            budget=self.token_budget,
        )
        prompt = (
            f"Session: {session_id}\n"
            f"User Input: {user_input}\n"
            f"{built.prompt}"
        )
        response = self.llm.generate(prompt)
        self.hierarchy.write_turn(session_id=session_id, user_input=user_input, agent_output=response)
        history.extend([f"user: {user_input}", f"assistant: {response}"])

        turn_count = len(history) // 2
        if turn_count % 20 == 0:
            self.hierarchy.consolidate_session(session_id=session_id, max_events=120, chunk_size=10)
            self.consolidation.run_full_consolidation()

        observations = [
            f"session_id={session_id}",
            f"context_tokens={built.total_tokens}",
            f"compressed={built.was_compressed}",
            f"turn_count={turn_count}",
        ]
        return ChatResult(
            session_id=session_id,
            response=response,
            context_tokens=built.total_tokens,
            was_compressed=built.was_compressed,
            observations=observations,
        )

    def analyze_long_document(
        self,
        session_id: str,
        title: str,
        document_text: str,
        question: str,
        target_tokens: int = 8000,
    ) -> ChatResult:
        compressed = self.long_context.compress(document_text, target_tokens=target_tokens, chunk_tokens=2000)
        self.ingest_document(session_id=session_id, title=title, text=compressed.compressed_text)
        result = self.chat(session_id=session_id, user_input=question)
        result.observations.append(f"document_original_tokens={compressed.original_tokens}")
        result.observations.append(f"document_compressed_tokens={compressed.compressed_tokens}")
        return result

    def get_session_history(self, session_id: str, limit: int = 50) -> list[MemoryRecord]:
        return self.memory_manager.read_session_history(session_id=session_id, limit=limit)
