import { render, screen } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";

import { StageDetailPanel } from "./StageDetailPanel";


vi.mock("@xyflow/react", () => ({
  Background: () => null,
  Controls: () => null,
  MarkerType: { ArrowClosed: "arrowclosed" },
  Position: { Left: "left", Right: "right" },
  ReactFlow: ({ nodes, edges }: { nodes: Array<{ position: { x: number } }>; edges: Array<{ label?: string }> }) => {
    const labels = edges.map((edge) => edge.label).filter(Boolean).join(", ");
    return <div data-testid="graph">{nodes.length} nodes, {edges.length} edges{labels ? `, ${labels}` : ""}; x={nodes.map((node) => node.position.x).join(",")}</div>;
  },
}));


describe("StageDetailPanel", () => {
  test("shows Stage 2 semantic and execution data", () => {
    render(<StageDetailPanel stageId="stage2" loading={false} result={{
      summary: { operator_count: 2, execution_group_count: 1 },
      graph: { nodes: [{ id: "matmul" }, { id: "softmax" }], edges: [{ source: "matmul", target: "softmax", kind: "data_dependency", tensor_ids: ["hidden"], label: "hidden [32x256]" }] },
      operators: [{ node_id: "matmul", kind: "tensor_contraction", semantic_op: "matmul", equation: "MK,KN->MN" }],
      execution_groups: [{ group_id: "g0", operator_ids: ["matmul", "softmax"], implementation: "unknown" }],
      warnings: ["softmax: no registered semantic normalizer"],
      raw_yaml: { operator_ir: "model_name: Mixed", execution_graph: "groups: []" },
    }} />);

    expect(screen.getByTestId("graph")).toHaveTextContent("2 nodes, 1 edges");
    expect(screen.getByTestId("graph")).toHaveTextContent("hidden [32x256]");
    expect(screen.getByTestId("graph")).toHaveTextContent("x=0,420");
    expect(screen.getByRole("heading", { name: "Execution groups" })).toBeInTheDocument();
    expect(screen.getByText("matmul, softmax")).toBeInTheDocument();
    expect(screen.getByText(/no registered semantic normalizer/)).toBeInTheDocument();
  });

  test("shows Stage 3 totals, dependencies and node costs", () => {
    render(<StageDetailPanel stageId="stage3" loading={false} result={{
      summary: { completeness: "complete", node_count: 1 },
      totals: { macs: 1024, flops: 2048, logical_read_bytes: 512, logical_write_bytes: 256 },
      graph: { nodes: [{ id: "matmul" }], edges: [] },
      nodes: [{ node_id: "matmul", operator_kind: "tensor_contraction", status: "supported", cost: { macs: 1024, flops: 2048 } }],
      raw_yaml: { cost_ir: "completeness: complete" },
    }} />);

    expect(screen.getByRole("heading", { name: "Node costs" })).toBeInTheDocument();
    expect(screen.getByText("tensor_contraction")).toBeInTheDocument();
    expect(screen.getByText(/1 nodes, 0 edges/)).toBeInTheDocument();
  });

  test("shows Stage 4 path, support state, assumptions and null components", () => {
    render(<StageDetailPanel stageId="stage4" loading={false} result={{
      summary: { hardware: "H100_PCIe", completeness: "complete" },
      components: { compute_lower_bound_us: 0.02, memory_lower_bound_us: 0.064, launch_us: null },
      diagnosis: { bottleneck: "memory", execution_path: "cuda_core+dram" },
      support: { status: "supported", confidence: "high", assumptions: ["dram_only_memory_path"], missing_capabilities: ["launch"] },
      raw_yaml: { performance_report: "hardware: H100_PCIe" },
    }} />);

    expect(screen.getByText("cuda_core+dram")).toBeInTheDocument();
    expect(screen.getByText("supported")).toBeInTheDocument();
    expect(screen.getByText("dram_only_memory_path")).toBeInTheDocument();
    expect(screen.getByText("launch")).toBeInTheDocument();
    expect(screen.getByText("Not supported")).toBeInTheDocument();
  });
});
