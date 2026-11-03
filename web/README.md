# SOLAR Four-Stage Web Demo

This isolated Web application runs SOLAR's existing four-stage offline analysis pipeline and exposes the actual stage state through a browser UI.

## Scope

- Input: a trusted PyTorch `Model.forward`, FP32 tensor shapes, and an allowlisted hardware configuration.
- Pipeline: Workload Graph, Hybrid Operator IR, Cost IR, and Performance prediction.
- Output: theoretical lower bound (`T_lb_us`), bottleneck, support state, and stage evidence.
- Runtime: CPU-only analysis. The selected target hardware is a performance model; no GPU or XPU workload is executed.

Submitted Python runs in a subprocess but is not sandboxed. Use this demo only with trusted code.

## Setup

From the repository root:

```bash
bash web/scripts/setup.sh
```

The default backend environment is `/Users/zhouzijian01/Desktop/workspace/pyvenv/solar-web`. Override paths with `SOLAR_PYVENV_ROOT`, `UV_BIN`, `UV_PROJECT_ENVIRONMENT`, or `UV_CACHE_DIR`. Local setup explicitly removes proxy environment variables.

## Run

```bash
bash web/scripts/dev.sh
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` and SSE traffic to FastAPI on `127.0.0.1:8000`.

Each task is stored at `web/.runtime/runs/<run_id>/`, including the request, generated model, `run_state.json`, four-stage YAML artifacts, and per-stage logs.

## Test

```bash
/Users/zhouzijian01/Desktop/workspace/pyvenv/solar-web/bin/python -m pytest -q web/backend/tests
npm test --prefix web/frontend
npm run build --prefix web/frontend
npm run e2e --prefix web/frontend
```

The integration and browser tests use the official FP32 Matmul workload with `H100_PCIe`. The accepted result is a memory-bound `T_lb_us` of approximately `0.0642887973`.

## Production Build Preview

```bash
bash web/scripts/build.sh
cd web/backend
/Users/zhouzijian01/Desktop/workspace/pyvenv/solar-web/bin/python -m uvicorn solar_web.main:app --host 127.0.0.1 --port 8000
```

After `frontend/dist` exists, FastAPI serves the built frontend at `http://127.0.0.1:8000`.
