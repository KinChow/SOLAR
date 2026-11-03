"""Versioned FastAPI routes."""

from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from solar_web.api.schemas import RunCreateRequest, RunCreateResponse
from solar_web.domain.models import StageId
from solar_web.services.input_builder import InputContractError
from solar_web.services.run_manager import (
    RunBusyError,
    RunManager,
    RunNotFoundError,
    StageResultUnavailableError,
    UnknownHardwareError,
)


router = APIRouter(prefix="/api/v1")


def _manager(request: Request) -> RunManager:
    return request.app.state.run_manager


@router.get("/hardware")
async def list_hardware(request: Request) -> list[dict[str, object]]:
    return _manager(request).list_hardware()


@router.post("/runs", response_model=RunCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_run(payload: RunCreateRequest, request: Request) -> RunCreateResponse:
    try:
        snapshot = await _manager(request).create_run(payload)
    except (UnknownHardwareError, InputContractError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RunBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return RunCreateResponse(run_id=snapshot.run_id)


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> dict[str, object]:
    try:
        return _manager(request).get_snapshot(run_id).model_dump(mode="json")
    except RunNotFoundError as error:
        raise HTTPException(status_code=404, detail="run not found") from error


@router.get("/runs/{run_id}/stages/{stage_id}")
async def get_stage(run_id: str, stage_id: StageId, request: Request) -> dict[str, object]:
    try:
        return _manager(request).get_stage_result(run_id, stage_id)
    except RunNotFoundError as error:
        raise HTTPException(status_code=404, detail="run not found") from error
    except StageResultUnavailableError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request) -> StreamingResponse:
    try:
        record = _manager(request).get_record(run_id)
    except RunNotFoundError as error:
        raise HTTPException(status_code=404, detail="run not found") from error

    async def events() -> AsyncIterator[str]:
        async for snapshot in record.events():
            yield f"event: snapshot\ndata: {snapshot.model_dump_json()}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
