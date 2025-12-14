"""LLM utilities for the Obsidian RAG agent."""

import asyncio
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .config import settings


def get_llm(cheap: bool = False, streaming: bool=False) -> ChatOpenAI:
    """Get LLM instance."""
    model = settings.openai_cheap_model if cheap else settings.openai_model
    return ChatOpenAI(
        model=model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        temperature=0,
        streaming=streaming,
    )


async def check_relevance(query: str, note_content: str, note_title: str) -> bool:
    """Check if a note is relevant to the query using a cheap LLM."""
    llm = get_llm(cheap=True, streaming=False)

    system_prompt = """You are a relevance checker. Your task is to determine if a note contains information relevant to answering a user's question.

Respond with ONLY "YES" if the note contains information that could help answer the question, or "NO" if it doesn't.

Be inclusive - if there's any chance the note might be useful, respond "YES"."""

    user_prompt = f"""Question: {query}

Note title: {note_title}

Note content:
{note_content[:2000]}  # Limit content to avoid token limits

Is this note relevant to answering the question? Respond with YES or NO only."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    answer = response.content.strip().upper()

    return answer.startswith("YES")


async def check_relevance_batch(
    query: str, notes: list[dict[str, Any]]
) -> list[bool]:
    """Check relevance for multiple notes in parallel."""
    tasks = [
        check_relevance(query, note["content"], note["title"]) for note in notes
    ]
    return await asyncio.gather(*tasks)


async def reformulate_query(original_query: str, history: list[dict] = None, writer=None) -> str:
    """Reformulate user query for better search results."""
    llm = get_llm(cheap=True, streaming=True)
    history = history or []

    system_prompt = """You are a query reformulator. Your task is to reformulate user questions to make them better for semantic search.

Rules:
1. Keep the core meaning intact
2. Expand abbreviations if any
3. Add relevant synonyms or related terms
4. Make the query more specific if it's too vague
5. If there is conversation history, use it for context to understand what the user is asking about
6. Return ONLY the reformulated query, nothing else"""

    history_text = ""
    if history:
        history_text = "\n\nConversation history:\n"
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"

    user_prompt = f"Reformulate this query for semantic search: {original_query}{history_text}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    return response.content.strip()


async def generate_answer(query: str, context: list[dict[str, Any]], history: list[dict] = None) -> str:
    """Generate final answer based on gathered context."""
    llm = get_llm(cheap=False, streaming=True)

    history = history or []

    # Format context
    context_text = ""
    for i, note in enumerate(context, 1):
        context_text += f"\n--- Note {i}: {note.get('title', 'Untitled')} ---\n"
        context_text += note.get("content", "")[:3000]
        context_text += "\n"

    system_prompt = r"""You are a helpful assistant that answers questions based on the provided notes from a knowledge base.
    GENERAL RULES:
    1. Answer based ONLY on the information in the provided notes.
    2. If the notes do not contain enough information, explicitly say so.
    3. Reference which notes you used when appropriate.
    4. Be concise but thorough.
    5. Use conversation history for coherent follow-up answers.
    6. Format the response using valid GitHub-Flavored Markdown (GFM).

    MATH & LATEX RULES (CRITICAL — KaTeX COMPATIBILITY):

    The output will be rendered using:
    - remark-math
    - rehype-katex
    - KaTeX (NOT full LaTeX)

    Therefore, you MUST follow these rules exactly:

    1. ALL mathematical content MUST be wrapped in dollar delimiters:
       - Inline math: $...$
       - Display (block) math: $$...$$

    2. NEVER output math outside of dollar delimiters.
       This includes:
       - Single variables ($x$, $y$, $t$)
       - Simple expressions ($x^2$, $\frac{a}{b}$)
       - Greek letters ($\alpha$, $\beta$, $\gamma$)
       - Operators ($\det$, $\sqrt{...}$)
       - Vectors and matrices ($\mathbf{x}$, $\Lambda$)

    3. STRICTLY FORBIDDEN syntax (DO NOT USE):
       - \( ... \)
       - \[ ... \]
       - \begin{equation} ... \end{equation}
       - \begin{equation*} ... \end{equation*}
       - \begin{align} ... \end{align}
       - \begin{align*} ... \end{align*}
       - \begin{gather} ... \end{gather}
       - \begin{gather*} ... \end{gather*}

    4. For multi-line or aligned equations, use ONLY KaTeX-safe environments,
       wrapped inside $$ ... $$:

       ✅ Correct:
       $$
       \begin{aligned}
       a &= b + c \\
       d &= e + f
       \end{aligned}
       $$

       ❌ Incorrect:
       \begin{align}
       a &= b + c
       \end{align}

    5. For matrices, use ONLY KaTeX-supported environments,
       always wrapped in $$ ... $$:

       - matrix
       - pmatrix
       - bmatrix
       - Bmatrix
       - vmatrix
       - Vmatrix
       - cases

       Example:
       $$
       \begin{pmatrix}
       a & b \\
       c & d
       \end{pmatrix}
       $$

    6. NEVER use LaTeX environments that KaTeX does not support.
       KaTeX is NOT full LaTeX — assume limited support.

    7. Do NOT attempt to auto-wrap math or guess delimiters.
       If something is mathematical, it MUST be explicitly written inside $ or $$.

    8. Use $$ ... $$ for:
       - standalone formulas
       - emphasized equations
       - multi-line expressions
       - matrices and cases

    9. Inline math ($...$) must stay inline and NOT span multiple lines.

    10. If you are unsure whether something should be math or plain text,
        prefer plain text and explain in words.

    CRITICAL FAILURE CONDITIONS:
    - Any mathematical expression outside of $ or $$ is considered INVALID output.
    - Any forbidden environment is considered INVALID output.
    - Such output may fail to render and must be avoided.

    Follow these rules strictly.
    """

    # Build history context
    history_text = ""
    if history:
        history_text = "\n\nConversation history:\n"
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n"
        history_text += "\n"

    user_prompt = f"""{history_text}Question: {query}

Context from knowledge base:
{context_text}

Please answer the question based on the context above."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    return response.content


async def should_extend_context(
    query: str, current_context: list[dict[str, Any]]
) -> bool:
    """Determine if we need to extend context to answer the query."""
    if not current_context:
        return True

    llm = get_llm(cheap=True, streaming=False)

    context_summary = "\n".join(
        f"- {note.get('title', 'Untitled')}: {note.get('content', '')[:500]}"
        for note in current_context[:5]
    )

    system_prompt = """You are an assistant that determines if gathered context is sufficient to answer a question.

Respond with ONLY "SUFFICIENT" if the context contains enough information to fully answer the question,
or "INSUFFICIENT" if more context is needed."""

    user_prompt = f"""Question: {query}

Current context (summaries):
{context_summary}

Is this context sufficient to answer the question? Respond with SUFFICIENT or INSUFFICIENT only."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = await llm.ainvoke(messages)
    answer = response.content.strip().upper()

    return not answer.startswith("SUFFICIENT")
