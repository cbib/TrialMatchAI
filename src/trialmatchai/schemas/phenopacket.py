from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Phenopacket(BaseModel):
    id: str = Field(..., min_length=1)
    metaData: Dict[str, Any]
    subject: Dict[str, Any]

    model_config = ConfigDict(extra="allow")


class Keywords(BaseModel):
    main_conditions: List[str] = Field(default_factory=list)
    other_conditions: List[str] = Field(default_factory=list)
    expanded_sentences: List[str] = Field(default_factory=list)
    error: Optional[str] = None

    model_config = ConfigDict(extra="allow")
