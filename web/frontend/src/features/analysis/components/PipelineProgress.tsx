import { Check, Circle, LoaderCircle, X } from "lucide-react";

import type { StageId, StageState } from "../types";


interface Props {
  stages: StageState[];
  selectedStage: StageId | null;
  onSelect: (stageId: StageId) => void;
}


const stageNumber: Record<StageId, number> = { stage1: 1, stage2: 2, stage3: 3, stage4: 4 };


function StatusIcon({ stage }: { stage: StageState }) {
  if (stage.status === "completed") return <Check size={17} />;
  if (stage.status === "running") return <LoaderCircle className="spin" size={17} />;
  if (stage.status === "failed") return <X size={17} />;
  return <Circle size={13} />;
}


export function PipelineProgress({ stages, selectedStage, onSelect }: Props) {
  return (
    <section className="pipeline" aria-label="Four-stage pipeline">
      {stages.map((stage, index) => (
        <div className="pipeline-segment" key={stage.stage_id}>
          <button
            type="button"
            aria-label={`Stage ${stageNumber[stage.stage_id]} ${stage.label}`}
            data-testid={`stage-${stage.stage_id}`}
            data-status={stage.status}
            className={`stage stage-${stage.status} ${selectedStage === stage.stage_id ? "selected" : ""}`}
            disabled={stage.status !== "completed"}
            onClick={() => onSelect(stage.stage_id)}
          >
            <span className="stage-icon"><StatusIcon stage={stage} /></span>
            <span><b>Stage {stageNumber[stage.stage_id]}</b><strong>{stage.label}</strong></span>
            <small>{stage.status}{stage.result_completeness && ` · ${stage.result_completeness}`}</small>
          </button>
          {index < stages.length - 1 && <div className={`connector ${stage.status === "completed" ? "complete" : ""}`}><span /></div>}
        </div>
      ))}
    </section>
  );
}
