import asyncio
from pathlib import Path

import httpx
import pytest

from solar_web.main import create_app


REPO_ROOT = Path(__file__).resolve().parents[3]
MATMUL_CODE = """class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 32, 64))

    def forward(self, x):
        return torch.matmul(self.weight, x)
"""


async def _wait_for_terminal(client: httpx.AsyncClient, run_id: str) -> dict:
    snapshot = None
    for _ in range(600):
        response = await client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200
        snapshot = response.json()
        if snapshot["status"] in {"completed", "failed"}:
            return snapshot
        await asyncio.sleep(0.05)
    raise AssertionError(f"run did not finish: {snapshot}")


@pytest.mark.asyncio
async def test_real_matmul_run_exposes_events_stage_results_and_final_result(tmp_path: Path) -> None:
    app = create_app(runtime_root=tmp_path, solar_root=REPO_ROOT, static_dir=None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        hardware = await client.get("/api/v1/hardware")
        assert hardware.status_code == 200
        assert {item["id"] for item in hardware.json()} >= {"H100_PCIe", "A6000"}

        created = await client.post(
            "/api/v1/runs",
            json={
                "operator_code": MATMUL_CODE,
                "inputs": [{"name": "x", "shape": [64, 128], "dtype": "float32"}],
                "hardware": "H100_PCIe",
            },
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]

        snapshot = await _wait_for_terminal(client, run_id)
        assert snapshot["status"] == "completed", snapshot
        assert [stage["status"] for stage in snapshot["stages"]] == ["completed"] * 4
        assert snapshot["final_result"]["theoretical_lower_bound_us"] == pytest.approx(
            0.06428879733176378,
            rel=1e-12,
        )
        assert snapshot["final_result"]["support_status"] == "supported"
        assert snapshot["final_result"]["bottleneck"] == "memory"

        stage = await client.get(f"/api/v1/runs/{run_id}/stages/stage2")
        assert stage.status_code == 200
        assert stage.json()["operators"][0]["kind"] == "tensor_contraction"

        events = await client.get(f"/api/v1/runs/{run_id}/events")
        assert events.status_code == 200
        assert "event: snapshot" in events.text
        assert '"status":"completed"' in events.text


@pytest.mark.asyncio
async def test_rejects_unknown_hardware_before_starting_run(tmp_path: Path) -> None:
    app = create_app(runtime_root=tmp_path, solar_root=REPO_ROOT, static_dir=None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/runs",
            json={
                "operator_code": MATMUL_CODE,
                "inputs": [{"name": "x", "shape": [64, 128], "dtype": "float32"}],
                "hardware": "../../private",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "unknown hardware configuration: ../../private"


@pytest.mark.asyncio
async def test_invalid_model_contract_is_reported_without_creating_run(tmp_path: Path) -> None:
    app = create_app(runtime_root=tmp_path, solar_root=REPO_ROOT, static_dir=None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/runs",
            json={
                "operator_code": "class Wrong: pass",
                "inputs": [{"name": "x", "shape": [16], "dtype": "float32"}],
                "hardware": "H100_PCIe",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "operator code must define class Model"


@pytest.mark.asyncio
async def test_stage_one_failure_stops_pipeline_and_preserves_log(tmp_path: Path) -> None:
    app = create_app(runtime_root=tmp_path, solar_root=REPO_ROOT, static_dir=None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/v1/runs",
            json={
                "operator_code": "class Model(nn.Module):\n    def forward(self, x):\n        raise RuntimeError('stage one failed')",
                "inputs": [{"name": "x", "shape": [16], "dtype": "float32"}],
                "hardware": "H100_PCIe",
            },
        )
        snapshot = await _wait_for_terminal(client, created.json()["run_id"])

    assert snapshot["status"] == "failed"
    assert [stage["status"] for stage in snapshot["stages"]] == ["failed", "skipped", "skipped", "skipped"]
    assert "failed to extract PyTorch graph" in snapshot["error"]
    assert (tmp_path / snapshot["run_id"] / "logs" / "stage1.log").is_file()


@pytest.mark.asyncio
async def test_unknown_operator_completes_with_explicit_unsupported_result(tmp_path: Path) -> None:
    app = create_app(runtime_root=tmp_path, solar_root=REPO_ROOT, static_dir=None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/v1/runs",
            json={
                "operator_code": "class Model(nn.Module):\n    def forward(self, x):\n        return torch.softmax(x, dim=-1)",
                "inputs": [{"name": "x", "shape": [2, 4, 128, 128], "dtype": "float32"}],
                "hardware": "H100_PCIe",
            },
        )
        snapshot = await _wait_for_terminal(client, created.json()["run_id"])

    assert snapshot["status"] == "completed"
    assert snapshot["stages"][1]["result_completeness"] == "partial"
    assert snapshot["stages"][2]["result_completeness"] == "partial"
    assert snapshot["final_result"]["support_status"] == "unsupported"
