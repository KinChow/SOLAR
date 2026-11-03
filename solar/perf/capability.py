# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardware capability models and legacy architecture adapters."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import yaml


_DTYPE_NAMES = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp64": "float64",
    "int8": "int8",
    "fp8": "float8",
    "tf32": "tf32",
    "nvfp4": "nvfp4",
}


@dataclass
class ComputeCapability:
    resource_id: str
    macs_per_cycle: Dict[str, float] = field(default_factory=dict)


@dataclass
class MemoryCapability:
    resource_id: str
    bytes_per_cycle: float
    capacity_bytes: int


@dataclass
class HardwareCapability:
    name: str
    vendor: str
    frequency_hz: float
    compute_resources: Dict[str, ComputeCapability]
    memory_resources: Dict[str, MemoryCapability]
    source: str


class HardwareCapabilityLoader:
    """Load a normalized capability from current Solar architecture YAML."""

    _MAC_KEY = re.compile(r"^MAC_per_cycle_(.+)_(tc|sm)$")

    def load(self, hardware: str) -> HardwareCapability:
        path = self._resolve(hardware)
        with path.open() as stream:
            data = yaml.safe_load(stream) or {}

        resources = {
            "tensor_core": ComputeCapability("tensor_core"),
            "cuda_core": ComputeCapability("cuda_core"),
        }
        for key, value in data.items():
            match = self._MAC_KEY.match(str(key))
            if not match:
                continue
            dtype_key, unit = match.groups()
            dtype = _DTYPE_NAMES.get(dtype_key, dtype_key)
            resource_id = "tensor_core" if unit == "tc" else "cuda_core"
            resources[resource_id].macs_per_cycle[dtype] = float(value)
        resources = {
            name: resource for name, resource in resources.items() if resource.macs_per_cycle
        }

        return HardwareCapability(
            name=str(data.get("name") or path.stem),
            vendor="nvidia",
            frequency_hz=float(data["freq_GHz"]) * 1e9,
            compute_resources=resources,
            memory_resources={
                "dram": MemoryCapability(
                    resource_id="dram",
                    bytes_per_cycle=float(data["DRAM_byte_per_cycle"]),
                    capacity_bytes=int(data.get("DRAM_capacity", 0)),
                )
            },
            source=str(path.resolve()),
        )

    @staticmethod
    def _resolve(hardware: str) -> Path:
        path = Path(hardware)
        if path.is_file():
            return path
        root = Path(__file__).resolve().parents[2]
        candidate = root / "configs" / "arch" / f"{hardware}.yaml"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"hardware configuration not found: {hardware}")


__all__ = [
    "ComputeCapability",
    "HardwareCapability",
    "HardwareCapabilityLoader",
    "MemoryCapability",
]
