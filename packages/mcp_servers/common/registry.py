"""Tool registry and unified execution."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from packages.authz.rbac import UserContext
from packages.authz.tool_policy import check_tool_permission
from packages.mcp_servers.common.errors import ToolNotFoundError
from packages.mcp_servers.common.permissions import list_allowed_tools
from packages.mcp_servers.common.schemas import ToolDefinition, ToolExecutionResult
from packages.mcp_servers.excel_report_mcp import tools as excel_tools
from packages.mcp_servers.excel_report_mcp.server import EXCEL_REPORT_TOOLS
from packages.mcp_servers.logistics_mcp import tools as logistics_tools
from packages.mcp_servers.logistics_mcp.server import LOGISTICS_TOOLS
from packages.mcp_servers.supplier_db_mcp import tools as supplier_tools
from packages.mcp_servers.supplier_db_mcp.server import SUPPLIER_DB_TOOLS
from packages.observability.repositories.tool_call_log_repo import write_tool_call_log

ToolHandler = Callable[[AsyncSession, UserContext, dict[str, Any]], Awaitable[dict[str, Any]]]

# 工具元数据注册表：用于对外展示工具能力、声明输入字段，并标记高风险操作。
# 这里只描述“工具是什么”，真正执行工具的函数由下方 TOOL_HANDLERS 绑定。
TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "query_supplier_profile": ToolDefinition(
        name="query_supplier_profile",
        description="Query supplier profile by supplier_code or supplier_id",
        server="supplier_db_mcp",
        input_schema={"supplier_code": "string", "supplier_id": "string"},
    ),
    "query_supplier_risk_events": ToolDefinition(
        name="query_supplier_risk_events",
        description="List recent risk events for a supplier",
        server="supplier_db_mcp",
        input_schema={"supplier_code": "string", "limit": "integer"},
    ),
    "query_supplier_delivery_history": ToolDefinition(
        name="query_supplier_delivery_history",
        description="List recent delivery orders for a supplier",
        server="supplier_db_mcp",
        input_schema={"supplier_code": "string", "limit": "integer"},
    ),
    "query_supplier_contract_summary": ToolDefinition(
        name="query_supplier_contract_summary",
        description="Summarize contract documents for a supplier",
        server="supplier_db_mcp",
        input_schema={"supplier_code": "string"},
    ),
    "query_delivery_status": ToolDefinition(
        name="query_delivery_status",
        description="Query delivery status by order_no or supplier_code",
        server="logistics_mcp",
        input_schema={"order_no": "string", "supplier_code": "string"},
    ),
    "query_delay_events": ToolDefinition(
        name="query_delay_events",
        description="List delivery delay events",
        server="logistics_mcp",
        input_schema={"supplier_code": "string"},
    ),
    "query_route_info": ToolDefinition(
        name="query_route_info",
        description="Query logistics route info",
        server="logistics_mcp",
        input_schema={"origin": "string", "destination": "string"},
    ),
    "export_risk_report": ToolDefinition(
        name="export_risk_report",
        description="Export risk report (approval required)",
        server="excel_report_mcp",
        input_schema={},
        high_risk=True,
    ),
    "export_benchmark_results": ToolDefinition(
        name="export_benchmark_results",
        description="Export benchmark results (approval required)",
        server="excel_report_mcp",
        input_schema={"benchmark_run_id": "string"},
        high_risk=True,
    ),
}

# 将工具名映射到具体 MCP 服务的处理函数。工具名必须与 TOOL_DEFINITIONS 保持一致，
# 这样统一执行入口才能在完成权限校验后找到实际业务实现。
TOOL_HANDLERS: dict[str, ToolHandler] = {
    "query_supplier_profile": supplier_tools.query_supplier_profile,
    "query_supplier_risk_events": supplier_tools.query_supplier_risk_events,
    "query_supplier_delivery_history": supplier_tools.query_supplier_delivery_history,
    "query_supplier_contract_summary": supplier_tools.query_supplier_contract_summary,
    "query_delivery_status": logistics_tools.query_delivery_status,
    "query_delay_events": logistics_tools.query_delay_events,
    "query_route_info": logistics_tools.query_route_info,
    "export_risk_report": excel_tools.export_risk_report,
    "export_benchmark_results": excel_tools.export_benchmark_results,
}


def get_tool_definitions_for_user(user_context: UserContext) -> list[ToolDefinition]:
    """返回当前用户按角色有权看到的工具定义。"""
    # 先由权限模块计算白名单，再与实际注册表求交集，避免返回未实现的工具。
    allowed = set(list_allowed_tools(user_context))
    return [TOOL_DEFINITIONS[name] for name in sorted(allowed) if name in TOOL_DEFINITIONS]


async def execute_tool_call(
    session: AsyncSession,
    user_context: UserContext,
    tool_name: str,
    tool_input: dict[str, Any] | None = None,
    *,
    trace_id: str | None = None,
    agent_run_id: uuid.UUID | str | None = None,
) -> ToolExecutionResult:
    """统一校验、执行工具并记录完整审计日志。"""
    # 复制输入，避免后续处理函数修改调用方持有的原始字典。
    tool_input = dict(tool_input or {})
    # trace_id 串联一次请求内的调用；tool_call_id 唯一标识本次工具执行。
    trace_id = trace_id or str(uuid.uuid4())
    tool_call_id = str(uuid.uuid4())
    started = time.perf_counter()

    # 未注册工具不会进入业务处理函数，但失败尝试仍写入日志，便于审计和排障。
    if tool_name not in TOOL_DEFINITIONS:
        result = ToolExecutionResult(
            tool_name=tool_name,
            status="failed",
            result={"error": f"unknown tool: {tool_name}"},
            tool_call_id=tool_call_id,
            trace_id=trace_id,
            permission_decision="denied",
            error_message=f"unknown tool: {tool_name}",
        )
        await write_tool_call_log(
            session,
            user_context=user_context,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=result.result,
            trace_id=trace_id,
            agent_run_id=agent_run_id,
            permission_decision="denied",
            approval_required=False,
            status="failed",
            latency_ms=0,
            error_message=result.error_message,
            tool_call_id=tool_call_id,
        )
        raise ToolNotFoundError(tool_name)

    # 在执行前校验角色权限、基础访问约束和高风险审批要求；需要数据库查询的
    # 供应商范围校验由具体工具处理函数继续完成。
    permission = check_tool_permission(user_context, tool_name, tool_input)
    if not permission.allowed:
        # 权限拒绝属于预期业务结果：记录原因后直接返回，不调用工具处理函数。
        latency_ms = int((time.perf_counter() - started) * 1000)
        result = ToolExecutionResult(
            tool_name=tool_name,
            status="denied",
            result={"error": permission.reason, "status": "denied"},
            tool_call_id=tool_call_id,
            trace_id=trace_id,
            permission_decision=permission.permission_decision,
            error_message=permission.reason,
            latency_ms=latency_ms,
        )
        await write_tool_call_log(
            session,
            user_context=user_context,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=result.result,
            trace_id=trace_id,
            agent_run_id=agent_run_id,
            permission_decision=permission.permission_decision,
            approval_required=False,
            status="denied",
            latency_ms=latency_ms,
            error_message=permission.reason,
            tool_call_id=tool_call_id,
        )
        return result

    if permission.approval_required:
        # 高风险工具在这里暂停，只返回待审批状态，不调用实际工具处理函数。
        latency_ms = int((time.perf_counter() - started) * 1000)
        result = ToolExecutionResult(
            tool_name=tool_name,
            status="approval_required",
            result={
                "status": "approval_required",
                "message": permission.reason,
                "tool_name": tool_name,
            },
            tool_call_id=tool_call_id,
            trace_id=trace_id,
            permission_decision=permission.permission_decision,
            approval_required=True,
            latency_ms=latency_ms,
        )
        await write_tool_call_log(
            session,
            user_context=user_context,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=result.result,
            trace_id=trace_id,
            agent_run_id=agent_run_id,
            permission_decision=permission.permission_decision,
            approval_required=True,
            status="approval_required",
            latency_ms=latency_ms,
            tool_call_id=tool_call_id,
        )
        return result

    # 权限通过后，才从处理函数注册表中取出实现并执行。
    handler = TOOL_HANDLERS[tool_name]
    try:
        output = await handler(session, user_context, tool_input)
        latency_ms = int((time.perf_counter() - started) * 1000)
        # 处理函数以结构化结果表达业务拒绝；其余正常输出视为成功。
        if output.get("status") == "denied" or output.get("error"):
            status = "denied"
        else:
            status = "success"
        result = ToolExecutionResult(
            tool_name=tool_name,
            status=status,
            result=output,
            tool_call_id=tool_call_id,
            trace_id=trace_id,
            permission_decision="allowed",
            latency_ms=latency_ms,
            error_message=output.get("error") if status != "success" else None,
        )
        await write_tool_call_log(
            session,
            user_context=user_context,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=output,
            trace_id=trace_id,
            agent_run_id=agent_run_id,
            permission_decision="allowed",
            approval_required=False,
            status=status,
            latency_ms=latency_ms,
            error_message=result.error_message,
            tool_call_id=tool_call_id,
        )
        return result
    except Exception as exc:
        # 捕获工具实现的运行时异常，将其转换为统一失败结果并保留审计记录。
        latency_ms = int((time.perf_counter() - started) * 1000)
        output = {"error": str(exc), "status": "failed"}
        result = ToolExecutionResult(
            tool_name=tool_name,
            status="failed",
            result=output,
            tool_call_id=tool_call_id,
            trace_id=trace_id,
            permission_decision="allowed",
            latency_ms=latency_ms,
            error_message=str(exc),
        )
        await write_tool_call_log(
            session,
            user_context=user_context,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=output,
            trace_id=trace_id,
            agent_run_id=agent_run_id,
            permission_decision="allowed",
            approval_required=False,
            status="failed",
            latency_ms=latency_ms,
            error_message=str(exc),
            tool_call_id=tool_call_id,
        )
        return result


def format_tool_evidence_content(tool_name: str, result: dict[str, Any]) -> str:
    """将工具结果序列化为可供证据聚合与引用的文本。"""
    return json.dumps({"tool": tool_name, "result": result}, ensure_ascii=False, indent=2)
