import { Background, Controls, MarkerType, Position, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { AlertTriangle, Braces, Cpu, Network, Route } from "lucide-react";
import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { StageId, StageResult } from "../types";


const titles: Record<StageId, string> = {
  stage1: "Stage 1 · Workload Graph",
  stage2: "Stage 2 · Hybrid Operator IR",
  stage3: "Stage 3 · Cost IR",
  stage4: "Stage 4 · Performance",
};


function Summary({ values }: { values: Record<string, unknown> }) {
  return <dl className="summary-list">{Object.entries(values).map(([name, value]) => <div key={name}><dt>{name.replaceAll("_", " ")}</dt><dd>{formatValue(value)}</dd></div>)}</dl>;
}


function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "Not supported";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "None";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}


function GraphView({ result }: { result: StageResult }) {
  if (!result.graph) return null;
  const ranks = new Map(result.graph.nodes.map((node) => [node.id, 0]));
  for (let pass = 0; pass < result.graph.nodes.length; pass += 1) {
    for (const edge of result.graph.edges) {
      const nextRank = (ranks.get(edge.source) ?? 0) + 1;
      if (nextRank > (ranks.get(edge.target) ?? 0)) ranks.set(edge.target, nextRank);
    }
  }
  const rankOffsets = new Map<number, number>();
  const rankSpacing = result.graph.edges.some((edge) => typeof edge.label === "string") ? 420 : 260;
  const nodes = result.graph.nodes.map((node) => {
    const rank = Math.min(ranks.get(node.id) ?? 0, result.graph!.nodes.length);
    const row = rankOffsets.get(rank) ?? 0;
    rankOffsets.set(rank, row + 1);
    return { id: node.id, data: { label: String(node.label ?? node.id) }, position: { x: rank * rankSpacing, y: row * 100 }, sourcePosition: Position.Right, targetPosition: Position.Left, type: "default" };
  });
  const edges = result.graph.edges.map((edge, index) => ({
    id: `e${index}`,
    source: edge.source,
    target: edge.target,
    label: typeof edge.label === "string" ? edge.label : undefined,
    markerEnd: { type: MarkerType.ArrowClosed },
  }));
  return <div className="graph-view"><ReactFlow nodes={nodes} edges={edges} fitView fitViewOptions={{ padding: 0.22 }}><Background /><Controls /></ReactFlow></div>;
}


function MetricChart({ values }: { values: Record<string, number | null> }) {
  const data = Object.entries(values).filter(([, value]) => typeof value === "number" && value > 0).map(([name, value]) => ({ name: name.replaceAll("_", " "), value }));
  if (!data.length) return <p className="empty-state">No numeric costs available.</p>;
  return <div className="metric-chart"><ResponsiveContainer width="100%" height={220}><BarChart data={data} layout="vertical" margin={{ left: 28 }}><CartesianGrid strokeDasharray="3 3" horizontal={false} /><XAxis type="number" /><YAxis dataKey="name" type="category" width={150} tick={{ fontSize: 11 }} /><Tooltip /><Bar dataKey="value" fill="#3867d6" radius={[0, 3, 3, 0]} /></BarChart></ResponsiveContainer></div>;
}


function ComponentBreakdown({ values }: { values: Record<string, number | null> }) {
  return <div className="component-breakdown"><Summary values={values} /><MetricChart values={values} /></div>;
}


function ExecutionGroups({ groups }: { groups: Array<Record<string, unknown>> }) {
  return <div className="data-section"><h3><Route size={16} /> Execution groups</h3><table><thead><tr><th>Group</th><th>Operators</th><th>Implementation</th><th>Fusion</th><th>Kernels</th></tr></thead><tbody>{groups.map((group) => <tr key={String(group.group_id)}><td>{String(group.group_id)}</td><td>{formatValue(group.operator_ids)}</td><td>{formatValue(group.implementation)}</td><td>{formatValue(group.fusion)}</td><td>{formatValue(group.kernel_count)}</td></tr>)}</tbody></table></div>;
}


function NodeCosts({ nodes }: { nodes: Array<Record<string, unknown>> }) {
  return <div className="data-section"><h3><Cpu size={16} /> Node costs</h3><table><thead><tr><th>Node</th><th>Kind</th><th>Status</th><th>Costs</th><th>Missing</th></tr></thead><tbody>{nodes.map((node) => <tr key={String(node.node_id)}><td>{String(node.node_id)}</td><td>{String(node.operator_kind)}</td><td>{String(node.status)}</td><td><code>{formatValue(node.cost)}</code></td><td>{formatValue(node.missing_costs)}</td></tr>)}</tbody></table></div>;
}


function PerformanceEvidence({ result }: { result: StageResult }) {
  const diagnosis = result.diagnosis ?? {};
  const support = result.support ?? {};
  return <div className="performance-evidence"><div className="data-section"><h3><Route size={16} /> Hardware mapping</h3><Summary values={diagnosis} /></div><div className="data-section"><h3>Support and assumptions</h3><Summary values={support} /></div></div>;
}


export function StageDetailPanel({ stageId, result, loading }: { stageId: StageId; result: StageResult | null; loading: boolean }) {
  const rawEntries = useMemo(() => Object.entries(result?.raw_yaml ?? {}), [result]);
  const [rawName, setRawName] = useState<string | null>(null);
  const selectedRaw = rawEntries.find(([name]) => name === rawName) ?? rawEntries[0];
  if (loading) return <section className="stage-detail loading">Loading stage result…</section>;
  if (!result) return null;
  return (
    <section className="stage-detail">
      <header><div><p className="eyebrow">Stage evidence</p><h2>{titles[stageId]}</h2></div>{result.warnings?.length ? <span className="warning-count"><AlertTriangle size={15} />{result.warnings.length} warning</span> : null}</header>
      {result.summary && <Summary values={result.summary} />}
      <GraphView result={result} />
      {stageId === "stage2" && result.operators && <div className="data-section"><h3><Network size={16} /> Operators</h3><table><thead><tr><th>Node</th><th>Kind</th><th>Semantic op</th><th>Equation</th><th>Dimensions</th></tr></thead><tbody>{result.operators.map((operator) => <tr key={String(operator.node_id)}><td>{String(operator.node_id)}</td><td>{String(operator.kind)}</td><td>{String(operator.semantic_op)}</td><td>{formatValue(operator.equation)}</td><td><code>{formatValue(operator.dimensions)}</code></td></tr>)}</tbody></table></div>}
      {stageId === "stage2" && result.execution_groups && <ExecutionGroups groups={result.execution_groups} />}
      {stageId === "stage3" && result.totals && <ComponentBreakdown values={result.totals} />}
      {stageId === "stage3" && result.nodes && <NodeCosts nodes={result.nodes} />}
      {stageId === "stage4" && result.components && <ComponentBreakdown values={result.components} />}
      {stageId === "stage4" && <PerformanceEvidence result={result} />}
      {stageId === "stage1" && result.tensors && <div className="data-section"><h3>Tensor table</h3><table><thead><tr><th>Tensor</th><th>Shape</th><th>Dtype</th><th>Role</th></tr></thead><tbody>{result.tensors.map((tensor) => <tr key={String(tensor.tensor_id)}><td>{String(tensor.tensor_id)}</td><td>{JSON.stringify(tensor.shape)}</td><td>{String(tensor.dtype)}</td><td>{String(tensor.role)}</td></tr>)}</tbody></table></div>}
      {result.warnings?.map((warning) => <div className="inline-warning" key={warning}><AlertTriangle size={15} />{warning}</div>)}
      {selectedRaw && <div className="raw-yaml"><div className="raw-tabs"><Braces size={15} />{rawEntries.map(([name]) => <button type="button" className={selectedRaw[0] === name ? "active" : ""} key={name} onClick={() => setRawName(name)}>{name}</button>)}</div><pre>{selectedRaw[1]}</pre></div>}
    </section>
  );
}
