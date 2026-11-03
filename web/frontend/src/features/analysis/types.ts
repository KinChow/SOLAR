export type RunStatus = "running" | "completed" | "failed";
export type StageStatus = "pending" | "running" | "completed" | "failed" | "skipped";
export type StageId = "stage1" | "stage2" | "stage3" | "stage4";

export interface TensorInput {
  name: string;
  shape: number[];
  dtype: "float32";
}

export interface RunCreateRequest {
  operator_code: string;
  inputs: TensorInput[];
  hardware: string;
}

export interface HardwareOption {
  id: string;
  name: string;
  validated: boolean;
}

export interface StageState {
  stage_id: StageId;
  label: string;
  status: StageStatus;
  started_at?: string | null;
  finished_at?: string | null;
  duration_ms?: number | null;
  result_completeness?: string | null;
  error?: string | null;
}

export interface FinalResult {
  theoretical_lower_bound_us: number | null;
  bottleneck: string | null;
  support_status: string | null;
  confidence: string | null;
}

export interface RunSnapshot {
  run_id: string;
  status: RunStatus;
  version: number;
  stages: StageState[];
  final_result?: FinalResult | null;
  error?: string | null;
}

export interface GraphData {
  nodes: Array<Record<string, unknown> & { id: string }>;
  edges: Array<Record<string, unknown> & { source: string; target: string }>;
}

export interface StageResult {
  summary?: Record<string, unknown>;
  graph?: GraphData;
  tensors?: Array<Record<string, unknown>>;
  operators?: Array<Record<string, unknown>>;
  execution_groups?: Array<Record<string, unknown>>;
  warnings?: string[];
  totals?: Record<string, number | null>;
  nodes?: Array<Record<string, unknown>>;
  dependencies?: Record<string, string[]>;
  components?: Record<string, number | null>;
  diagnosis?: Record<string, unknown>;
  support?: Record<string, unknown>;
  raw_yaml?: Record<string, string>;
}
