# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAML serialization for versioned IR models."""

from pathlib import Path
from typing import Type, TypeVar

import yaml

from solar.common.utils import NoAliasDumper


SchemaT = TypeVar("SchemaT")


def dump_ir(model: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = model.to_dict()  # type: ignore[attr-defined]
    with path.open("w") as stream:
        yaml.dump(data, stream, Dumper=NoAliasDumper, sort_keys=False)


def load_ir(model_type: Type[SchemaT], path: Path) -> SchemaT:
    with path.open() as stream:
        data = yaml.safe_load(stream) or {}
    return model_type.from_dict(data)  # type: ignore[attr-defined,no-any-return]
