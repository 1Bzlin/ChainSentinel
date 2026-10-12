"""Execute the validated task DAG."""

from __future__ import annotations

from packages.agents.execution import execute_plan
from packages.agents.planning.schemas import task_summary
from packages.agents.state import AgentState
from packages.observability.repositories.execution_plan_repo import (
    finalize_execution_plan,
)


async def plan_executor_node(state: AgentState) -> AgentState:
    # The AgentRun and plan audit rows must be visible to independent task sessions.
    await state["db_session"].commit()
    plan, results = await execute_plan(
        state["execution_plan"],
        user_context=state["user_context"],
        trace_id=state["trace_id"],
        agent_run_id=state.get("agent_run_id"),
        chunk_size_config=state.get("chunk_size_config", 512),
        ip_address=state.get("ip_address"),
    )
    await finalize_execution_plan(state["db_session"], plan)
    by_id = {result.task_id: result for result in results}
    return {
        "execution_plan": plan,
        "plan_status": plan.status,
        "task_results": results,
        "plan_summary": [
            task_summary(task, by_id.get(task.task_id)) for task in plan.tasks
        ],
        "nodes_executed": [
            "plan_executor",
            *(f"task:{result.task_type}:{result.task_id}" for result in results),
        ],
    }
