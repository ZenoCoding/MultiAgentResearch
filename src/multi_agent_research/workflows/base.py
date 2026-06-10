from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha256
import json
from typing import Any

from multi_agent_research.context import RunContext
from multi_agent_research.models import PromptTemplate, TaskInput, WorkflowSpec


class Workflow(ABC):
    name: str
    version: str

    @abstractmethod
    async def run(self, task: TaskInput, context: RunContext) -> str:
        raise NotImplementedError

    @abstractmethod
    def config(self) -> dict[str, Any]:
        raise NotImplementedError

    def prompt_templates(self) -> list[PromptTemplate]:
        return []

    def spec(self) -> WorkflowSpec:
        config = self.config()
        prompts = self.prompt_templates()
        identity = {
            "schema_version": "1",
            "name": self.name,
            "version": self.version,
            "config": config,
            "prompts": [prompt.model_dump() for prompt in prompts],
        }
        canonical = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return WorkflowSpec(
            name=self.name,
            version=self.version,
            fingerprint=sha256(canonical.encode("utf-8")).hexdigest(),
            config=config,
            prompts=prompts,
        )
