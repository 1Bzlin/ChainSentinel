"use client";

import { useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import Link from "next/link";
import {
  ReactFlow,
  Background,
  Controls,
  Node,
  Edge,
  MarkerType,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { PageHeader } from "@/components/layout/PageHeader";
import { SectionCard } from "@/components/common/SectionCard";
import { TraceIdBadge } from "@/components/badges/TraceIdBadge";
import { chatQuery, getTrace } from "@/lib/api";

const WORKFLOW_NODES = [
  "permission", "context_loader", "planner", "plan_validator", "clarify",
  "plan_executor", "evidence_aggregator", "report", "review", "memory_writer",
];

const initialNodes: Node[] = WORKFLOW_NODES.map((name, i) => ({
  id: name,
  position: { x: (i % 3) * 220, y: Math.floor(i / 3) * 100 },
  data: { label: name.replace(/_/g, " ") },
  style: {
    background: "#f8fafc",
    border: "1px solid #e2e8f0",
    borderRadius: 8,
    padding: 8,
    fontSize: 11,
    minWidth: 140,
  },
}));

const initialEdges: Edge[] = WORKFLOW_NODES.slice(0, -1).map((n, i) => ({
  id: `e-${n}`,
  source: n,
  target: WORKFLOW_NODES[i + 1],
  markerEnd: { type: MarkerType.ArrowClosed },
  style: { stroke: "#cbd5e1" },
}));

export default function AgentPage() {
  const [query, setQuery] = useState("Public Supply Policy 的主要内容是什么？");
  const [traceId, setTraceId] = useState("");
  const [nodeDetails, setNodeDetails] = useState<Record<string, unknown> | null>(null);
  const [sessionId, setSessionId] = useState("");

  useEffect(() => {
    const existing = window.sessionStorage.getItem("supplyguard-agent-session");
    const next = existing || crypto.randomUUID();
    window.sessionStorage.setItem("supplyguard-agent-session", next);
    setSessionId(next);
  }, []);

  const startNewSession = () => {
    const next = crypto.randomUUID();
    window.sessionStorage.setItem("supplyguard-agent-session", next);
    setSessionId(next);
    setTraceId("");
  };

  const chat = useMutation({
    mutationFn: () => chatQuery({ query, session_id: sessionId, use_memory: true }),
    onSuccess: (data) => setTraceId(data.trace_id),
  });

  const loadTrace = useMutation({
    mutationFn: () => getTrace(traceId),
  });

  const executed = chat.data?.nodes_executed || [];
  const nodes = initialNodes.map((n) => ({
    ...n,
    style: {
      ...n.style,
      background: executed.includes(n.id) ? "#eef2ff" : executed.includes(n.id.replace("_skipped", "")) ? "#fef3c7" : "#f8fafc",
      border: executed.includes(n.id) ? "2px solid #6366f1" : "1px solid #e2e8f0",
    },
  }));

  return (
    <div>
      <PageHeader title="Agent Workflow" description="LangGraph harness — Permission → Planner → Multi-task execution → Report → Review" />
      <div className="grid gap-6 lg:grid-cols-3">
        <SectionCard title="Run Agent">
          <textarea value={query} onChange={(e) => setQuery(e.target.value)} rows={3} className="w-full rounded-lg border p-2 text-sm" />
          <div className="mt-2 flex items-center justify-between gap-2 text-[11px] text-slate-500">
            <span className="truncate font-mono" title={sessionId}>session: {sessionId || "initializing"}</span>
            <button type="button" onClick={startNewSession} className="shrink-0 rounded border px-2 py-1">New session</button>
          </div>
          <button type="button" onClick={() => chat.mutate()} disabled={chat.isPending || !sessionId} className="mt-3 w-full rounded-lg bg-brand-600 py-2 text-sm text-white disabled:opacity-50">
            {chat.isPending ? "Running…" : "Run Chat Workflow"}
          </button>
          {chat.data && (
            <div className="mt-4 space-y-2 text-sm">
              <p>Plan: <strong>{chat.data.plan_status}</strong></p>
              <p>Planning: <strong>{chat.data.planning_source}</strong> · confidence {chat.data.planning_confidence?.toFixed(2) ?? "-"}</p>
              {chat.data.resolved_query && chat.data.resolved_query !== query && <p className="text-xs text-slate-500">Resolved: {chat.data.resolved_query}</p>}
              {chat.data.plan_summary.length > 0 && (
                <div className="mt-2 space-y-1 text-xs text-slate-600">
                  {chat.data.plan_summary.map((task) => (
                    <p key={task.task_id}>
                      {task.task_id} · {task.task_type} · {task.status}
                    </p>
                  ))}
                </div>
              )}
              <Link href={`/traces?trace=${chat.data.trace_id}`}><TraceIdBadge traceId={chat.data.trace_id} /></Link>
              <pre className="text-xs bg-slate-50 p-2 rounded max-h-40 overflow-auto whitespace-pre-wrap">{chat.data.answer}</pre>
            </div>
          )}
          <div className="mt-4">
            <input value={traceId} onChange={(e) => setTraceId(e.target.value)} placeholder="trace_id" className="w-full rounded border p-2 text-xs font-mono" />
            <button type="button" onClick={() => loadTrace.mutate()} disabled={!traceId} className="mt-2 w-full rounded border py-1.5 text-xs">Load trace</button>
          </div>
        </SectionCard>
        <div className="lg:col-span-2">
          <div className="h-[420px] rounded-xl border bg-white">
            <ReactFlow nodes={nodes} edges={initialEdges} fitView onNodeClick={(_, n) => setNodeDetails({ id: n.id, executed: executed.includes(n.id) })}>
              <Background />
              <Controls />
            </ReactFlow>
          </div>
          {loadTrace.data && (
            <SectionCard title="Trace Node Spans" className="mt-4">
              <pre className="text-xs overflow-auto max-h-48">{JSON.stringify(loadTrace.data.node_spans, null, 2)}</pre>
            </SectionCard>
          )}
          {nodeDetails && (
            <p className="mt-2 text-xs text-slate-500">Selected: {String(nodeDetails.id)} · executed: {String(nodeDetails.executed)}</p>
          )}
        </div>
      </div>
    </div>
  );
}
