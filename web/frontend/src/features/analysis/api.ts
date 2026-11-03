import type { HardwareOption, RunCreateRequest, RunSnapshot, StageId, StageResult } from "./types";


async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: unknown } | null;
    throw new Error(errorDetail(payload?.detail) ?? `Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}


function errorDetail(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (!Array.isArray(detail)) return null;
  const messages = detail.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const validation = item as { loc?: unknown; msg?: unknown };
    if (typeof validation.msg !== "string") return [];
    const location = Array.isArray(validation.loc)
      ? validation.loc.filter((part) => part !== "body").join(".")
      : "";
    return [`${location ? `${location}: ` : ""}${validation.msg}`];
  });
  return messages.length ? messages.join("; ") : null;
}


export function listHardware(): Promise<HardwareOption[]> {
  return request("/api/v1/hardware");
}


export function createRun(payload: RunCreateRequest): Promise<{ run_id: string }> {
  return request("/api/v1/runs", { method: "POST", body: JSON.stringify(payload) });
}


export function getRun(runId: string): Promise<RunSnapshot> {
  return request(`/api/v1/runs/${runId}`);
}


export function getStageResult(runId: string, stageId: StageId): Promise<StageResult> {
  return request(`/api/v1/runs/${runId}/stages/${stageId}`);
}
