"""Execute existing four-stage CLIs and update observable run state."""

import sys
from pathlib import Path
from typing import List, Sequence, Tuple

from solar_web.api.schemas import RunCreateRequest
from solar_web.domain.models import RunRecord, StageId
from solar_web.services.artifact_reader import ArtifactReader
from solar_web.workers.command_runner import CommandRunner


class PipelineWorker:
    def __init__(self, runner: CommandRunner | None = None, timeout_seconds: float = 60) -> None:
        self.runner = runner or CommandRunner()
        self.timeout_seconds = timeout_seconds

    async def run(
        self,
        record: RunRecord,
        request: RunCreateRequest,
        model_file: Path,
    ) -> None:
        reader = ArtifactReader(record.run_dir)
        stages = self._commands(record.run_dir, model_file, request.hardware, request.case_id)
        for index, (stage_id, command) in enumerate(stages):
            try:
                result = await self.runner.run(
                    command,
                    log_path=record.run_dir / "logs" / f"{stage_id.value}.log",
                    timeout_seconds=self.timeout_seconds,
                )
                if result.returncode != 0:
                    detail = result.stderr.strip() or result.stdout.strip() or "stage process failed"
                    await record.fail_stage(stage_id, detail[-4_000:])
                    return
                completeness = reader.result_completeness(stage_id)
                await record.complete_stage(
                    stage_id,
                    result_completeness=completeness,
                    duration_ms=result.duration_ms,
                )
            except Exception as error:
                await record.fail_stage(stage_id, str(error))
                return

            if index + 1 < len(stages):
                await record.start_stage(stages[index + 1][0])

        await record.complete_run(reader.read_final_result())

    @staticmethod
    def _commands(
        run_dir: Path,
        model_file: Path,
        hardware: str,
        case_id: str = "default",
    ) -> List[Tuple[StageId, Sequence[str]]]:
        stage1 = run_dir / "stage1"
        stage2 = run_dir / "stage2"
        stage3 = run_dir / "stage3"
        stage4 = run_dir / "stage4"
        return [
            (
                StageId.WORKLOAD,
                [
                    sys.executable,
                    "-m",
                    "solar.cli.extract_workload",
                    "--model-file",
                    str(model_file),
                    "--output-dir",
                    str(stage1),
                    "--case-id",
                    case_id,
                    "--force-rerun",
                ],
            ),
            (
                StageId.OPERATOR_IR,
                [
                    sys.executable,
                    "-m",
                    "solar.cli.normalize_operators",
                    "--workload-graph",
                    str(stage1 / "workload_graph.yaml"),
                    "--output-dir",
                    str(stage2),
                ],
            ),
            (
                StageId.COST,
                [
                    sys.executable,
                    "-m",
                    "solar.cli.analyze_cost",
                    "--operator-ir",
                    str(stage2 / "operator_ir.yaml"),
                    "--execution-graph",
                    str(stage2 / "execution_graph.yaml"),
                    "--output-dir",
                    str(stage3),
                ],
            ),
            (
                StageId.PERFORMANCE,
                [
                    sys.executable,
                    "-m",
                    "solar.cli.predict_performance",
                    "--cost-ir",
                    str(stage3 / "cost_ir.yaml"),
                    "--execution-graph",
                    str(stage2 / "execution_graph.yaml"),
                    "--output-dir",
                    str(stage4),
                    "--hardware",
                    hardware,
                ],
            ),
        ]
