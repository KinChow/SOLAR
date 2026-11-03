import { lazy, Suspense, useCallback, useEffect, useState } from "react";

import * as api from "./api";
import { FinalResultPanel } from "./components/FinalResultPanel";
import { PipelineProgress } from "./components/PipelineProgress";
import { SubmissionForm } from "./components/SubmissionForm";
import type { HardwareOption, RunCreateRequest, RunSnapshot, StageId, StageResult } from "./types";
import { useRunEvents } from "./useRunEvents";


const StageDetailPanel = lazy(() => import("./components/StageDetailPanel").then((module) => ({ default: module.StageDetailPanel })));


function initialSnapshot(runId: string): RunSnapshot {
  return {
    run_id: runId,
    status: "running",
    version: 0,
    stages: [
      { stage_id: "stage1", label: "Workload Graph", status: "running" },
      { stage_id: "stage2", label: "Hybrid Operator IR", status: "pending" },
      { stage_id: "stage3", label: "Cost IR", status: "pending" },
      { stage_id: "stage4", label: "Performance", status: "pending" },
    ],
  };
}


export function AnalysisPage() {
  const [hardware, setHardware] = useState<HardwareOption[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<RunSnapshot | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedStage, setSelectedStage] = useState<StageId | null>(null);
  const [stageResult, setStageResult] = useState<StageResult | null>(null);
  const [stageLoading, setStageLoading] = useState(false);

  useEffect(() => {
    api.listHardware().then(setHardware).catch((loadError) => setError(loadError.message));
  }, []);

  const receiveSnapshot = useCallback((next: RunSnapshot) => {
    setSnapshot(next);
    if (next.status === "failed") setError(next.error ?? "Analysis failed");
  }, []);
  const eventError = useCallback(() => {
    // EventSource reconnects automatically. The last valid snapshot remains visible.
  }, []);
  useRunEvents(runId, receiveSnapshot, eventError);

  const submit = async (request: RunCreateRequest) => {
    setSubmitting(true);
    setError(null);
    setSelectedStage(null);
    setStageResult(null);
    try {
      const created = await api.createRun(request);
      setRunId(created.run_id);
      setSnapshot(initialSnapshot(created.run_id));
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Unable to start analysis");
    } finally {
      setSubmitting(false);
    }
  };

  const selectStage = async (stageId: StageId) => {
    if (!runId) return;
    setSelectedStage(stageId);
    setStageResult(null);
    setStageLoading(true);
    try {
      setStageResult(await api.getStageResult(runId, stageId));
    } catch (stageError) {
      setError(stageError instanceof Error ? stageError.message : "Unable to load stage result");
    } finally {
      setStageLoading(false);
    }
  };

  if (!snapshot) {
    return <SubmissionForm hardware={hardware} submitting={submitting} error={error} onSubmit={submit} />;
  }

  return (
    <main className="run-shell">
      <header className="run-header">
        <div><p className="eyebrow">Analysis run</p><h1>{snapshot.status === "completed" ? "Analysis complete" : snapshot.status === "failed" ? "Analysis failed" : "Pipeline running"}</h1></div>
        <div className={`run-status status-${snapshot.status}`}>{snapshot.status}</div>
      </header>
      {snapshot.status === "completed" && snapshot.final_result && <FinalResultPanel result={snapshot.final_result} />}
      <PipelineProgress stages={snapshot.stages} selectedStage={selectedStage} onSelect={selectStage} />
      {snapshot.status === "running" && <div className="live-note"><span className="live-dot" /> Pipeline state is updating from the analysis backend.</div>}
      {snapshot.status === "failed" && <div className="run-error" role="alert">{snapshot.error}</div>}
      {selectedStage && <Suspense fallback={<section className="stage-detail loading">Loading stage view…</section>}><StageDetailPanel key={selectedStage} stageId={selectedStage} result={stageResult} loading={stageLoading} /></Suspense>}
    </main>
  );
}
