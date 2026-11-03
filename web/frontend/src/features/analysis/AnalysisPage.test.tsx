import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { AnalysisPage } from "./AnalysisPage";
import * as api from "./api";
import type { RunSnapshot } from "./types";


vi.mock("@uiw/react-codemirror", () => ({
  default: ({ value, onChange }: { value: string; onChange: (value: string) => void }) => (
    <textarea aria-label="Operator code" value={value} onChange={(event) => onChange(event.target.value)} />
  ),
}));

vi.mock("@xyflow/react", () => ({
  Background: () => null,
  Controls: () => null,
  Position: { Left: "left", Right: "right" },
  ReactFlow: ({ nodes }: { nodes: Array<{ id: string }> }) => <div data-testid="graph">{nodes.length} nodes</div>,
}));


class FakeEventSource {
  static latest: FakeEventSource;
  private listeners = new Map<string, (event: MessageEvent) => void>();

  constructor(public url: string) {
    FakeEventSource.latest = this;
  }

  addEventListener(name: string, listener: EventListener) {
    this.listeners.set(name, listener as (event: MessageEvent) => void);
  }

  removeEventListener(name: string) {
    this.listeners.delete(name);
  }

  close() {}

  emit(snapshot: RunSnapshot) {
    this.listeners.get("snapshot")?.({ data: JSON.stringify(snapshot) } as MessageEvent);
  }
}


const completedSnapshot: RunSnapshot = {
  run_id: "run-1",
  status: "completed",
  version: 8,
  stages: [
    { stage_id: "stage1", label: "Workload Graph", status: "completed", result_completeness: "complete" },
    { stage_id: "stage2", label: "Hybrid Operator IR", status: "completed", result_completeness: "complete" },
    { stage_id: "stage3", label: "Cost IR", status: "completed", result_completeness: "complete" },
    { stage_id: "stage4", label: "Performance", status: "completed", result_completeness: "complete" },
  ],
  final_result: {
    theoretical_lower_bound_us: 0.06428879733176378,
    bottleneck: "memory",
    support_status: "supported",
    confidence: "high",
  },
};


describe("AnalysisPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.stubGlobal("EventSource", FakeEventSource);
    vi.spyOn(api, "listHardware").mockResolvedValue([
      { id: "H100_PCIe", name: "H100_PCIe", validated: true },
      { id: "A6000", name: "A6000", validated: false },
    ]);
    vi.spyOn(api, "createRun").mockResolvedValue({ run_id: "run-1" });
    vi.spyOn(api, "getStageResult").mockResolvedValue({
      summary: { model_name: "Matmul", operator_count: 1, tensor_count: 3 },
      graph: { nodes: [], edges: [] },
      tensors: [],
      raw_yaml: { workload_graph: "model_name: Matmul" },
    });
  });

  test("shows only code, inputs and hardware before submission", async () => {
    render(<AnalysisPage />);

    expect(screen.getByRole("heading", { name: "New analysis" })).toBeInTheDocument();
    expect((screen.getByLabelText("Operator code") as HTMLTextAreaElement).value).toContain("self.w3");
    expect(screen.getByLabelText("Input shape")).toHaveValue("32, 128");
    expect(await screen.findByRole("option", { name: "H100_PCIe · validated" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "A6000 · unverified" })).toBeInTheDocument();
    expect(screen.queryByText("Workload Graph")).not.toBeInTheDocument();
  });

  test("shows all stages immediately and advances only from SSE snapshots", async () => {
    const user = userEvent.setup();
    render(<AnalysisPage />);
    await screen.findByRole("option", { name: "H100_PCIe · validated" });

    await user.click(screen.getByRole("button", { name: "Run analysis" }));

    expect(await screen.findByText("Workload Graph")).toBeInTheDocument();
    expect(screen.getByText("Hybrid Operator IR")).toBeInTheDocument();
    expect(screen.getByText("Cost IR")).toBeInTheDocument();
    expect(screen.getByText("Performance")).toBeInTheDocument();
    expect(screen.getByTestId("stage-stage1")).toHaveAttribute("data-status", "running");
    expect(screen.getByTestId("stage-stage2")).toHaveAttribute("data-status", "pending");
    expect(screen.queryByText("Theoretical lower bound")).not.toBeInTheDocument();

    FakeEventSource.latest.emit(completedSnapshot);

    expect(await screen.findByText("Theoretical lower bound")).toBeInTheDocument();
    expect(screen.getByText("0.0643 µs")).toBeInTheDocument();
    expect(screen.getByText("Memory bound")).toBeInTheDocument();
  });

  test("allows completed stages to open details", async () => {
    const user = userEvent.setup();
    render(<AnalysisPage />);
    await screen.findByRole("option", { name: "H100_PCIe · validated" });
    await user.click(screen.getByRole("button", { name: "Run analysis" }));
    FakeEventSource.latest.emit(completedSnapshot);

    await user.click(await screen.findByRole("button", { name: /Stage 1 Workload Graph/ }));

    await waitFor(() => expect(api.getStageResult).toHaveBeenCalledWith("run-1", "stage1"));
    expect(await screen.findByRole("heading", { name: "Stage 1 · Workload Graph" }, { timeout: 5_000 })).toBeInTheDocument();
    expect(screen.getByText("model_name: Matmul")).toBeInTheDocument();
  });

  test("preserves completed evidence and skips later stages after failure", async () => {
    const user = userEvent.setup();
    render(<AnalysisPage />);
    await screen.findByRole("option", { name: "H100_PCIe · validated" });
    await user.click(screen.getByRole("button", { name: "Run analysis" }));

    FakeEventSource.latest.emit({
      run_id: "run-1",
      status: "failed",
      version: 4,
      stages: [
        { stage_id: "stage1", label: "Workload Graph", status: "completed", result_completeness: "complete" },
        { stage_id: "stage2", label: "Hybrid Operator IR", status: "failed", error: "normalization failed" },
        { stage_id: "stage3", label: "Cost IR", status: "skipped" },
        { stage_id: "stage4", label: "Performance", status: "skipped" },
      ],
      error: "normalization failed",
    });

    expect(await screen.findByRole("heading", { name: "Analysis failed" })).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("normalization failed");
    expect(screen.getByTestId("stage-stage3")).toHaveAttribute("data-status", "skipped");
    expect(screen.getByTestId("stage-stage4")).toBeDisabled();
    await user.click(screen.getByTestId("stage-stage1"));
    expect(await screen.findByRole("heading", { name: "Stage 1 · Workload Graph" })).toBeInTheDocument();
  });
});
