"""Deterministic validation and authorization for generated plans."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.agents.planning.entity_extractor import (
    ENTITY_PATTERNS,
    extract_query_entities,
    normalize_supplier_code,
)
from packages.agents.planning.schemas import (
    ExecutionPlan,
    GraphTask,
    MemoryTask,
    RagTask,
    ToolTask,
)
from packages.authz.rbac import UserContext
from packages.authz.tool_policy import check_tool_permission
from packages.common.db.models.supplier import Supplier
from packages.graphrag.graph_permissions import check_graph_query_permission
from packages.graphrag.graph_schema import QUERY_TYPES
from packages.mcp_servers.common.registry import TOOL_DEFINITIONS


def _has_cycle(tasks) -> bool:
    graph = {task.task_id: task.depends_on for task in tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> bool:
        if task_id in visiting:
            return True
        if task_id in visited:
            return False
        visiting.add(task_id)
        for dependency in graph.get(task_id, []):
            if visit(dependency):
                return True
        visiting.remove(task_id)
        visited.add(task_id)
        return False

    return any(visit(task_id) for task_id in graph)


def _block(plan: ExecutionPlan, error: str) -> ExecutionPlan:
    plan.status = "blocked"
    plan.source = "guard"
    plan.validation_errors.append(error)
    return plan


def _clarify(plan: ExecutionPlan, error: str, question: str) -> ExecutionPlan:
    plan.status = "clarification_required"
    plan.source = "guard"
    plan.tasks = []
    plan.validation_errors.append(error)
    plan.clarification_question = question
    return plan


def _missing_tool_inputs(task: ToolTask) -> list[str]:
    values = task.tool_input
    if task.tool_name in {
        "query_supplier_risk_events",
        "query_supplier_delivery_history",
        "query_supplier_contract_summary",
        "query_delay_events",
    }:
        return [] if values.get("supplier_code") else ["supplier_code"]
    if task.tool_name == "query_supplier_profile":
        return (
            []
            if values.get("supplier_code") or values.get("supplier_id")
            else ["supplier_code"]
        )
    if task.tool_name == "query_delivery_status":
        return (
            []
            if values.get("order_no") or values.get("supplier_code")
            else ["order_no or supplier_code"]
        )
    if task.tool_name == "query_route_info":
        return [key for key in ("origin", "destination") if not values.get(key)]
    if task.tool_name == "export_benchmark_results":
        return [] if values.get("benchmark_run_id") else ["benchmark_run_id"]
    return []


def _graph_entity_matches(task: GraphTask) -> bool:
    entity_types = {
        "supplier_impact": "supplier",
        "supplier_alternatives": "supplier",
        "delivery_delay_impact": "supplier",
        "product_dependency": "product",
        "customer_order_dependency": "customer",
        "quality_issue_trace": "quality",
    }
    entity_type = entity_types.get(task.graph_query_type)
    return bool(
        entity_type and ENTITY_PATTERNS[entity_type].fullmatch(task.entity_code.upper())
    )


async def validate_execution_plan(
    session: AsyncSession,
    user_context: UserContext,
    plan: ExecutionPlan,
    *,
    session_id: str,
    use_memory: bool,
) -> ExecutionPlan:
    if plan.status == "clarification_required":
        if not plan.clarification_question:
            plan.clarification_question = "请再明确一下希望查询的对象和分析内容。"
        return plan
    if plan.confidence < 0.55:
        return _clarify(
            plan,
            "planning confidence below 0.55",
            plan.clarification_question or "请再明确一下希望查询的内容。",
        )
    if not plan.tasks:
        return _clarify(plan, "empty executable plan", "请明确需要完成的查询或分析。")

    if not use_memory:
        disabled_memory_ids = {
            task.task_id for task in plan.tasks if isinstance(task, MemoryTask)
        }
        if disabled_memory_ids:
            if any(
                disabled_memory_ids.intersection(task.depends_on)
                for task in plan.tasks
                if not isinstance(task, MemoryTask)
            ):
                return _clarify(
                    plan,
                    "an executable task depends on disabled memory",
                    "该请求依赖历史记忆，请启用记忆后重试或补充完整上下文。",
                )
            plan.validation_errors.append(
                "memory tasks removed because memory is disabled"
            )
            plan.tasks = [
                task for task in plan.tasks if not isinstance(task, MemoryTask)
            ]
            if not plan.tasks:
                return _clarify(
                    plan,
                    "memory is disabled",
                    "当前请求需要历史记忆，请启用记忆或提供完整上下文。",
                )

    task_ids = [task.task_id for task in plan.tasks]
    if len(task_ids) != len(set(task_ids)):
        return _clarify(
            plan, "duplicate task ids", "请求计划存在冲突，请换一种方式描述。"
        )
    known_ids = set(task_ids)
    if any(
        dependency not in known_ids
        for task in plan.tasks
        for dependency in task.depends_on
    ):
        return _clarify(
            plan, "unknown task dependency", "请求中的执行顺序不明确，请重新描述。"
        )
    if _has_cycle(plan.tasks):
        return _clarify(
            plan, "cyclic task dependency", "请求包含循环依赖，请拆分后重试。"
        )

    allowed_entities = {
        *extract_query_entities(plan.original_query).values(),
        *plan.entities.values(),
    }
    for task in plan.tasks:
        candidate_entities: list[str] = []
        if isinstance(task, RagTask) and task.supplier_code:
            candidate_entities.append(task.supplier_code)
        elif isinstance(task, GraphTask):
            candidate_entities.append(task.entity_code)
        elif isinstance(task, ToolTask):
            candidate_entities.extend(
                str(task.tool_input[key])
                for key in ("supplier_code", "order_no")
                if task.tool_input.get(key)
            )
        for candidate in candidate_entities:
            normalized = (
                normalize_supplier_code(candidate)
                if candidate.upper().startswith(("SUP-", "S"))
                else candidate.upper()
            )
            if normalized not in allowed_entities:
                return _clarify(
                    plan,
                    f"task {task.task_id} introduced entity {normalized}",
                    f"计划中出现了问题或上下文未提供的实体 {normalized}，请确认查询对象。",
                )

    supplier_codes: set[str] = set()
    for task in plan.tasks:
        if isinstance(task, RagTask) and task.supplier_code:
            supplier_codes.add(normalize_supplier_code(task.supplier_code))
        elif isinstance(task, GraphTask) and task.entity_code.upper().startswith(
            ("SUP-", "S")
        ):
            supplier_codes.add(normalize_supplier_code(task.entity_code))
        elif isinstance(task, ToolTask) and task.tool_input.get("supplier_code"):
            supplier_codes.add(
                normalize_supplier_code(str(task.tool_input["supplier_code"]))
            )

    suppliers: dict[str, Supplier] = {}
    if supplier_codes:
        rows = (
            (
                await session.execute(
                    select(Supplier).where(
                        Supplier.tenant_id == user_context.tenant_id,
                        Supplier.code.in_(supplier_codes),
                    )
                )
            )
            .scalars()
            .all()
        )
        suppliers = {row.code: row for row in rows}
        missing = supplier_codes - set(suppliers)
        if missing:
            code = sorted(missing)[0]
            return _clarify(
                plan,
                f"supplier not found: {code}",
                f"当前租户中没有找到供应商 {code}，请确认供应商编码。",
            )

    own_supplier: Supplier | None = None
    if user_context.is_external_supplier:
        own_supplier = (
            await session.execute(
                select(Supplier).where(
                    Supplier.tenant_id == user_context.tenant_id,
                    Supplier.id == user_context.supplier_id,
                )
            )
        ).scalar_one_or_none()
        if own_supplier is None:
            return _block(plan, "external supplier binding not found")
        if supplier_codes and supplier_codes != {own_supplier.code}:
            return _block(plan, "external supplier cross-supplier request")

    retained = []
    denied_ids: set[str] = set()
    for task in plan.tasks:
        denial: str | None = None
        if isinstance(task, MemoryTask):
            task.session_id = session_id
        elif isinstance(task, RagTask):
            if task.supplier_id and not task.supplier_code:
                return _clarify(
                    plan,
                    f"task {task.task_id} supplied an unverified supplier_id",
                    "请使用供应商编码指定文档检索范围。",
                )
            task.supplier_id = None
            code = (
                normalize_supplier_code(task.supplier_code)
                if task.supplier_code
                else None
            )
            if own_supplier:
                code = own_supplier.code
            if code:
                task.supplier_code = code
                task.supplier_id = str(suppliers.get(code, own_supplier).id)
        elif isinstance(task, GraphTask):
            if task.graph_query_type not in QUERY_TYPES:
                denial = f"unsupported graph query type: {task.graph_query_type}"
            elif not _graph_entity_matches(task):
                return _clarify(
                    plan,
                    f"entity {task.entity_code} does not match {task.graph_query_type}",
                    "图分析类型与实体编码不匹配，请确认需要分析的对象。",
                )
            supplier_code = None
            if task.entity_code.upper().startswith(("SUP-", "S")):
                supplier_code = normalize_supplier_code(task.entity_code)
                task.entity_code = supplier_code
            if own_supplier and not supplier_code:
                supplier_code = own_supplier.code
            if denial is None:
                permission = check_graph_query_permission(
                    user_context,
                    task.graph_query_type,
                    supplier_code=supplier_code,
                    resolved_supplier_scope=own_supplier.code if own_supplier else None,
                )
                if not permission.allowed:
                    denial = permission.reason
        elif isinstance(task, ToolTask):
            if task.tool_input.get("supplier_id"):
                return _clarify(
                    plan,
                    f"task {task.task_id} supplied an unverified supplier_id",
                    "请使用供应商编码调用供应商查询工具。",
                )
            benchmark_run_id = task.tool_input.get("benchmark_run_id")
            if benchmark_run_id and str(benchmark_run_id) not in plan.original_query:
                return _clarify(
                    plan,
                    f"task {task.task_id} introduced benchmark_run_id",
                    "请提供需要导出的评测运行 ID。",
                )
            if task.tool_name not in TOOL_DEFINITIONS:
                denial = f"unknown tool: {task.tool_name}"
            else:
                if own_supplier and "supplier_code" not in task.tool_input:
                    task.tool_input["supplier_code"] = own_supplier.code
                missing = _missing_tool_inputs(task)
                if missing:
                    return _clarify(
                        plan,
                        f"task {task.task_id} missing inputs: {', '.join(missing)}",
                        f"执行 {task.tool_name} 还需要参数：{', '.join(missing)}。",
                    )
                permission = check_tool_permission(
                    user_context, task.tool_name, task.tool_input
                )
                if not permission.allowed:
                    denial = permission.reason

        if denial:
            plan.validation_errors.append(f"{task.task_id}: {denial}")
            denied_ids.add(task.task_id)
            if task.required:
                return _block(plan, f"required task denied: {task.task_id}: {denial}")
        else:
            retained.append(task)

    # A non-required task depending on a removed task cannot be executed either.
    changed = True
    while changed:
        changed = False
        next_retained = []
        for task in retained:
            if denied_ids.intersection(task.depends_on):
                denied_ids.add(task.task_id)
                plan.validation_errors.append(f"{task.task_id}: dependency denied")
                changed = True
            else:
                next_retained.append(task)
        retained = next_retained

    plan.tasks = retained
    if not retained:
        return _block(plan, "no authorized tasks remain")
    return plan
