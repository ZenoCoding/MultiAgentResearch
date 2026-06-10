from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from string import Template
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class HarnessModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextContent(HarnessModel):
    type: Literal["text"] = "text"
    text: str


class ImageURL(HarnessModel):
    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class ImageContent(HarnessModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


ContentPart = Annotated[
    TextContent | ImageContent,
    Field(discriminator="type"),
]


class Message(HarnessModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart]
    name: str | None = None


class PromptReference(HarnessModel):
    name: str
    version: str
    content_sha256: str


class PromptTemplate(HarnessModel):
    name: str
    version: str
    template: str
    content_sha256: str = ""

    @model_validator(mode="after")
    def set_or_validate_hash(self) -> PromptTemplate:
        expected = sha256(self.template.encode("utf-8")).hexdigest()
        if self.content_sha256 and self.content_sha256 != expected:
            raise ValueError("content_sha256 does not match prompt template")
        self.content_sha256 = expected
        return self

    def reference(self) -> PromptReference:
        return PromptReference(
            name=self.name,
            version=self.version,
            content_sha256=self.content_sha256,
        )

    def render(self, **values: Any) -> str:
        return Template(self.template).substitute(
            {key: str(value) for key, value in values.items()}
        )


class TaskSource(HarnessModel):
    benchmark: str
    version: str | None = None
    split: str | None = None
    original_id: str | None = None


class AnswerChoice(HarnessModel):
    label: str
    text: str | None = None


class AnswerSpec(HarnessModel):
    type: Literal[
        "free_text",
        "short_answer",
        "multiple_choice",
        "number",
        "json",
        "code",
    ] = "short_answer"
    choices: list[AnswerChoice] = Field(default_factory=list)
    include_explanation: bool = True
    include_confidence: bool = False
    custom_instruction: str | None = None

    @model_validator(mode="after")
    def validate_choices(self) -> AnswerSpec:
        if self.type == "multiple_choice" and not self.choices:
            raise ValueError("multiple_choice answers require choices")
        labels = [choice.label for choice in self.choices]
        if len(labels) != len(set(labels)):
            raise ValueError("answer choice labels must be unique")
        return self

    def instruction(self) -> str:
        instructions: list[str] = []
        if self.custom_instruction:
            instructions.append(self.custom_instruction.strip())
        if self.type == "multiple_choice":
            labels = ", ".join(choice.label for choice in self.choices)
            instructions.append(f"The final answer must be one of: {labels}.")
        elif self.type == "number":
            instructions.append("The final answer must be a number.")
        elif self.type == "json":
            instructions.append("The final answer must be valid JSON.")
        elif self.type == "code":
            instructions.append("The final answer must contain the requested code.")

        if self.include_explanation:
            instructions.append(
                "You may explain your reasoning before the final answer."
            )
        else:
            instructions.append("Do not include an explanation.")

        instructions.append(
            "End your response with exactly one <final_answer>...</final_answer> block."
        )
        if self.include_confidence:
            instructions.append(
                "After the final answer, include <confidence>0-100</confidence>."
            )
        return "\n".join(instructions)

    def prompt_reference(self) -> PromptReference:
        instruction = self.instruction()
        return PromptReference(
            name="task.answer_contract",
            version="1.0.0",
            content_sha256=sha256(instruction.encode("utf-8")).hexdigest(),
        )


class TaskInput(HarnessModel):
    """Gold-free, model-visible benchmark task."""

    id: str
    messages: list[Message]
    answer_spec: AnswerSpec = Field(default_factory=AnswerSpec)
    source: TaskSource | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_messages(self) -> TaskInput:
        if not self.messages:
            raise ValueError("benchmark tasks require at least one message")
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("benchmark tasks require at least one user message")
        return self

    @classmethod
    def from_prompt(
        cls,
        *,
        id: str,
        prompt: str,
        answer_spec: AnswerSpec | None = None,
        source: TaskSource | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskInput:
        return cls(
            id=id,
            messages=[Message(role="user", content=prompt)],
            answer_spec=answer_spec or AnswerSpec(),
            source=source,
            metadata=metadata or {},
        )


class AgentSpec(HarnessModel):
    id: str
    model: str
    system_prompt: str = ""
    system_prompt_name: str | None = None
    system_prompt_version: str = "inline"
    reasoning_effort: str | None = None
    service_tier: Literal["auto", "default", "flex", "priority"] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_first_class_parameters(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        parameters = dict(normalized.get("parameters") or {})
        nested_effort = parameters.pop("reasoning_effort", None)
        nested_service_tier = parameters.pop("service_tier", None)
        explicit_effort = normalized.get("reasoning_effort")
        explicit_service_tier = normalized.get("service_tier")
        if (
            nested_effort is not None
            and explicit_effort is not None
            and nested_effort != explicit_effort
        ):
            raise ValueError(
                "reasoning_effort conflicts with parameters.reasoning_effort"
            )
        if explicit_effort is None and nested_effort is not None:
            normalized["reasoning_effort"] = nested_effort
        if (
            nested_service_tier is not None
            and explicit_service_tier is not None
            and nested_service_tier != explicit_service_tier
        ):
            raise ValueError("service_tier conflicts with parameters.service_tier")
        if explicit_service_tier is None and nested_service_tier is not None:
            normalized["service_tier"] = nested_service_tier
        normalized["parameters"] = parameters
        return normalized

    def completion_parameters(self) -> dict[str, Any]:
        parameters = dict(self.parameters)
        if self.reasoning_effort is not None:
            parameters["reasoning_effort"] = self.reasoning_effort
        if self.service_tier is not None:
            parameters["service_tier"] = self.service_tier
        return parameters


class UsageStats(HarnessModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None


class CallError(HarnessModel):
    type: str
    message: str
    traceback: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ModelCallRecord(HarnessModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    sequence: int
    run_id: str
    task_id: str
    workflow: str
    step: str
    agent_id: str
    requested_model: str
    request_parameters: dict[str, Any] = Field(default_factory=dict)
    response_model: str | None = None
    response_service_tier: str | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    messages: list[Message]
    prompt_references: list[PromptReference] = Field(default_factory=list)
    output: Message | None = None
    usage: UsageStats = Field(default_factory=UsageStats)
    cost_usd: float | None = None
    started_at: datetime
    ended_at: datetime
    latency_ms: float
    status: Literal["success", "failed"]
    error: CallError | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] | None = None


class WorkflowEvent(HarnessModel):
    type: str
    timestamp: datetime = Field(default_factory=utc_now)
    data: dict[str, Any] = Field(default_factory=dict)


class WorkflowSpec(HarnessModel):
    schema_version: str = "1"
    name: str
    version: str
    fingerprint: str
    config: dict[str, Any] = Field(default_factory=dict)
    prompts: list[PromptTemplate] = Field(default_factory=list)


class GitProvenance(HarnessModel):
    repository_root: str
    commit: str | None = None
    branch: str | None = None
    dirty: bool
    status_short: str
    remote_url: str | None = None


class RunProvenance(HarnessModel):
    schema_version: str = "1"
    working_directory: str
    argv: list[str]
    python_version: str
    python_implementation: str
    python_executable: str
    platform: str
    machine: str
    harness_version: str | None = None
    litellm_version: str | None = None
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)
    credential_fingerprints: dict[str, str] = Field(default_factory=dict)
    secret_file_sha256: dict[str, str] = Field(default_factory=dict)
    lockfile_sha256: str | None = None
    source_snapshot_sha256: str
    source_files: dict[str, str] = Field(default_factory=dict)
    git: GitProvenance | None = None


class RunRequest(HarnessModel):
    id: str
    experiment_id: str
    created_at: datetime = Field(default_factory=utc_now)
    task: TaskInput
    workflow: WorkflowSpec
    provenance: RunProvenance


class RunMetrics(HarnessModel):
    model_calls: int = 0
    failed_model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0
    summed_call_latency_ms: float = 0.0
    wall_time_ms: float = 0.0

    @classmethod
    def from_calls(
        cls, calls: list[ModelCallRecord], wall_time_ms: float
    ) -> RunMetrics:
        return cls(
            model_calls=len(calls),
            failed_model_calls=sum(call.status == "failed" for call in calls),
            input_tokens=sum(call.usage.input_tokens or 0 for call in calls),
            output_tokens=sum(call.usage.output_tokens or 0 for call in calls),
            total_tokens=sum(call.usage.total_tokens or 0 for call in calls),
            reasoning_tokens=sum(call.usage.reasoning_tokens or 0 for call in calls),
            cached_input_tokens=sum(
                call.usage.cached_input_tokens or 0 for call in calls
            ),
            cost_usd=sum(call.cost_usd or 0.0 for call in calls),
            summed_call_latency_ms=sum(call.latency_ms for call in calls),
            wall_time_ms=wall_time_ms,
        )


class WorkflowOutput(HarnessModel):
    raw_response: str
    answer: str
    confidence: float | None = None
    parse_status: Literal["parsed", "fallback"]
    contract_valid: bool
    validation_errors: list[str] = Field(default_factory=list)

    @classmethod
    def from_response(
        cls,
        response: str,
        answer_spec: AnswerSpec,
    ) -> WorkflowOutput:
        matches = re.findall(
            r"<final_answer>\s*(.*?)\s*</final_answer>",
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if matches:
            answer = matches[-1].strip()
            parse_status: Literal["parsed", "fallback"] = "parsed"
        else:
            answer = response.strip()
            parse_status = "fallback"

        confidence: float | None = None
        if answer_spec.include_confidence:
            confidence_matches = re.findall(
                r"<confidence>\s*([0-9]+(?:\.[0-9]+)?)\s*</confidence>",
                response,
                flags=re.IGNORECASE,
            )
            if confidence_matches:
                confidence = min(100.0, max(0.0, float(confidence_matches[-1])))

        validation_errors: list[str] = []
        if parse_status == "fallback":
            validation_errors.append("missing final_answer block")
        if not answer:
            validation_errors.append("answer is empty")
        if answer_spec.include_confidence and confidence is None:
            validation_errors.append("missing or invalid confidence block")
        if answer_spec.type == "multiple_choice":
            labels = {choice.label.casefold() for choice in answer_spec.choices}
            if answer.casefold() not in labels:
                validation_errors.append("answer is not an allowed choice label")
        elif answer_spec.type == "number":
            try:
                float(answer.replace(",", ""))
            except ValueError:
                validation_errors.append("answer is not a number")
        elif answer_spec.type == "json":
            try:
                json.loads(answer)
            except json.JSONDecodeError:
                validation_errors.append("answer is not valid JSON")

        return cls(
            raw_response=response,
            answer=answer,
            confidence=confidence,
            parse_status=parse_status,
            contract_valid=not validation_errors,
            validation_errors=validation_errors,
        )


class StageAnswer(HarnessModel):
    sequence: int
    step: str
    kind: Literal["candidate", "aggregate"]
    agent_id: str | None = None
    call_id: str | None = None
    output: WorkflowOutput
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunResult(HarnessModel):
    run_id: str
    experiment_id: str
    task_id: str
    workflow: WorkflowSpec
    provenance: RunProvenance
    status: Literal["success", "failed"]
    final_answer: str | None = None
    output: WorkflowOutput | None = None
    error: CallError | None = None
    started_at: datetime
    ended_at: datetime
    metrics: RunMetrics
    calls: list[ModelCallRecord]
    stage_answers: list[StageAnswer] = Field(default_factory=list)
    events: list[WorkflowEvent]
