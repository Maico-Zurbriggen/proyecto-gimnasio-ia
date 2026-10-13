"""Export the versioned persistence contract; contains no data or credentials."""

import json
from pathlib import Path
from typing import Annotated

from pydantic import Field, TypeAdapter

from gym_engine.contracts import RoutineProposalV2, RoutineUnableV2

schema = TypeAdapter(
    Annotated[RoutineProposalV2 | RoutineUnableV2, Field(discriminator="outcome")]
).json_schema()
schema["$id"] = "urn:gym:routine-generation:2.0"
schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
target = Path(__file__).resolve().parents[1] / "openapi" / "routine-generation-v2.schema.json"
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
