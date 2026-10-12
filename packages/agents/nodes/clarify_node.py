"""Return a focused clarification instead of guessing a route or entity."""

from packages.agents.state import AgentState


async def clarify_node(state: AgentState) -> AgentState:
    question = (
        state.get("clarification_question") or "请再明确一下你希望分析的对象和目标。"
    )
    return {
        "draft_answer": question,
        "citations": [],
        "token_usage": {},
        "status": "clarification_required",
        "nodes_executed": ["clarify"],
    }
