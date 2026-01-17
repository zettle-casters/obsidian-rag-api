"""LangGraph agent for Obsidian RAG with recursive context extension and multi-vault support."""

import asyncio
import operator
from typing import Annotated, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from obsidian_retriever.manager import KnowledgeBaseManager

from ..infrastructure.config import settings
from .vault_manager import get_vault_manager
from ..infrastructure.llm import (
    reformulate_query,
    generate_answer,
    should_extend_context,
    check_relevance_batch,
)


class NoteContext(TypedDict):
    """Context from a single note."""

    note_id: str
    title: str
    content: str
    depth: int


class AgentState(TypedDict):
    """State for the RAG agent."""

    # Input
    original_query: str
    vault_id: str
    model_name: str | None

    # Conversation history
    messages: Annotated[list[dict], operator.add]

    # Working state
    reformulated_query: str
    search_results: list[dict]
    knowledge_base: Annotated[list[NoteContext], operator.add]
    notes_to_explore: list[str]
    explored_notes: set[str]
    current_depth: int

    # Output
    final_answer: str
    status: str


async def reformulate_node(state: AgentState) -> dict:
    """Reformulate the user query for better search."""
    original = state["original_query"]

    writer = get_stream_writer()

    history = state.get("messages", [])

    writer("🔄 Начинаю переформулировку запроса...")

    reformulated = await reformulate_query(original, history, writer, state.get("model_name"))

    new_messages = [{"role": "user", "content": original}]

    writer("✅ Переформулировка завершена")

    return {
        "reformulated_query": reformulated,
        "explored_notes": set(),
        "knowledge_base": [],
        "current_depth": 0,
        "messages": new_messages,
    }


async def search_node(state: AgentState) -> dict:
    """Search for relevant notes using the retriever."""
    vault_id = state["vault_id"]
    retriever = get_vault_manager(vault_id)

    if not retriever:
        return {
            "search_results": [],
            "notes_to_explore": [],
            "knowledge_base": [],
            "explored_notes": set(),
            "status": "error",
        }

    query = state["reformulated_query"] or state["original_query"]

    results = retriever.search_notes(query, top_k=settings.search_top_k)

    search_results = []
    notes_to_explore = []

    for note_record, score in results:
        # Get full note content
        content_parts = []
        if note_record.chunk_ids:
            for chunk_id in note_record.chunk_ids:
                chunk = retriever.get_chunk(chunk_id)
                if chunk:
                    content_parts.append(chunk.text)

        note_data = {
            "note_id": note_record.note_id,
            "title": note_record.title,
            "content": "\n\n".join(content_parts),
            "score": score,
            "linked_notes": note_record.links_to_notes or [],
        }
        search_results.append(note_data)
        notes_to_explore.append(note_record.note_id)

    # Add initial results to knowledge base
    initial_knowledge = [
        NoteContext(
            note_id=r["note_id"],
            title=r["title"],
            content=r["content"],
            depth=0,
        )
        for r in search_results
    ]

    return {
        "search_results": search_results,
        "notes_to_explore": notes_to_explore,
        "knowledge_base": initial_knowledge,
        "explored_notes": set(notes_to_explore),
    }


async def check_context_node(state: AgentState) -> dict:
    """Check if current context is sufficient."""
    query = state["reformulated_query"] or state["original_query"]
    knowledge = state["knowledge_base"]

    needs_more = await should_extend_context(query, knowledge, state.get("model_name"))

    if needs_more and state["current_depth"] < settings.max_recursion_depth:
        return {"status": "extend"}
    else:
        return {"status": "generate"}


async def extend_context_node(state: AgentState) -> dict:
    """Extend context by exploring linked notes."""
    vault_id = state["vault_id"]
    retriever = get_vault_manager(vault_id)

    if not retriever:
        return {
            "status": "generate",
            "current_depth": state["current_depth"] + 1,
        }

    query = state["reformulated_query"] or state["original_query"]
    explored = state["explored_notes"]
    current_depth = state["current_depth"]

    # Collect all linked notes from current knowledge base
    all_linked_ids = set()
    for note in state["knowledge_base"]:
        note_record = retriever.get_note(note["note_id"])
        if note_record and note_record.links_to_notes:
            for linked_id in note_record.links_to_notes:
                if linked_id not in explored:
                    all_linked_ids.add(linked_id)

        # Also get incoming links
        outgoing, incoming = retriever.get_note_neighbors(note["note_id"])
        for link in outgoing:
            if link.to_note_id not in explored:
                all_linked_ids.add(link.to_note_id)
        for link in incoming:
            if link.from_note_id not in explored:
                all_linked_ids.add(link.from_note_id)

    if not all_linked_ids:
        return {
            "status": "generate",
            "current_depth": current_depth + 1,
        }

    # Get content for all linked notes
    linked_notes_data = []
    for note_id in all_linked_ids:
        note = retriever.get_note(note_id)
        if note:
            content_parts = []
            if note.chunk_ids:
                for chunk_id in note.chunk_ids:
                    chunk = retriever.get_chunk(chunk_id)
                    if chunk:
                        content_parts.append(chunk.text)

            linked_notes_data.append(
                {
                    "note_id": note.note_id,
                    "title": note.title,
                    "content": "\n\n".join(content_parts),
                }
            )

    # Check relevance in parallel
    relevance_results = await check_relevance_batch(query, linked_notes_data, state.get("model_name"))

    # Add relevant notes to knowledge base
    new_knowledge = []
    new_explored = set(explored)

    for note_data, is_relevant in zip(linked_notes_data, relevance_results):
        new_explored.add(note_data["note_id"])
        if is_relevant:
            new_knowledge.append(
                NoteContext(
                    note_id=note_data["note_id"],
                    title=note_data["title"],
                    content=note_data["content"],
                    depth=current_depth + 1,
                )
            )

    return {
        "knowledge_base": new_knowledge,
        "explored_notes": new_explored,
        "current_depth": current_depth + 1,
    }


async def generate_answer_node(state: AgentState) -> dict:
    """Generate final answer from accumulated knowledge."""
    query = state["original_query"]
    knowledge = state["knowledge_base"]
    history = state.get("messages", [])

    # Deduplicate by note_id, keeping lowest depth
    seen = {}
    for note in knowledge:
        note_id = note["note_id"]
        if note_id not in seen or note["depth"] < seen[note_id]["depth"]:
            seen[note_id] = note

    unique_knowledge = list(seen.values())

    if not unique_knowledge:
        answer = "I couldn't find any relevant information in the knowledge base to answer your question."
    else:
        answer = await generate_answer(query, unique_knowledge, history, state.get("model_name"))

    # Add assistant response to history
    new_messages = [{"role": "assistant", "content": answer}]

    return {
        "final_answer": answer,
        "status": "complete",
        "messages": new_messages,
    }


def route_after_check(state: AgentState) -> str:
    """Route based on context check result."""
    if state["status"] == "extend":
        return "extend_context"
    return "generate_answer"


def route_after_extend(state: AgentState) -> str:
    """Route after extending context."""
    if state["current_depth"] >= settings.max_recursion_depth:
        return "generate_answer"
    return "check_context"


def build_graph() -> StateGraph:
    """Build the LangGraph agent."""
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("reformulate", reformulate_node)
    builder.add_node("search", search_node)
    builder.add_node("check_context", check_context_node)
    builder.add_node("extend_context", extend_context_node)
    builder.add_node("generate_answer", generate_answer_node)

    # Add edges
    builder.add_edge(START, "reformulate")
    builder.add_edge("reformulate", "search")
    builder.add_edge("search", "check_context")
    builder.add_conditional_edges(
        "check_context",
        route_after_check,
        {"extend_context": "extend_context", "generate_answer": "generate_answer"},
    )
    builder.add_conditional_edges(
        "extend_context",
        route_after_extend,
        {"check_context": "check_context", "generate_answer": "generate_answer"},
    )
    builder.add_edge("generate_answer", END)

    return builder


def create_agent():
    """Create the compiled agent with memory."""
    graph = build_graph()
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Global agent instances per vault
_agents: dict[str, any] = {}


def get_agent_for_vault(vault_id: str):
    """Get or create agent instance for a specific vault."""
    if vault_id not in _agents:
        _agents[vault_id] = create_agent()
    return _agents[vault_id]


async def run_agent_with_vault(
    query: str,
    vault_id: str,
    thread_id: str = "default",
    model_name: str | None = None,
) -> dict:
    """Run the agent with a query for a specific vault."""
    agent = get_agent_for_vault(vault_id)

    initial_state = {
        "original_query": query,
        "vault_id": vault_id,
        "model_name": model_name,
        "messages": [],
        "reformulated_query": "",
        "search_results": [],
        "knowledge_base": [],
        "notes_to_explore": [],
        "explored_notes": set(),
        "current_depth": 0,
        "final_answer": "",
        "status": "started",
    }

    config = {"configurable": {"thread_id": thread_id}}

    result = await agent.ainvoke(initial_state, config)

    return {
        "query": query,
        "reformulated_query": result.get("reformulated_query", ""),
        "answer": result.get("final_answer", ""),
        "notes_used": len(result.get("knowledge_base", [])),
        "max_depth_reached": result.get("current_depth", 0),
    }
