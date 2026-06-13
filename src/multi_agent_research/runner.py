from __future__ import annotations

from time import monotonic
import traceback
from typing import Any
from uuid import uuid4

from multi_agent_research.aggregation import AggregationInconclusive
from multi_agent_research.context import RunContext
from multi_agent_research.llm import LLMClient
from multi_agent_research.models import (
    CallError,
    InconclusiveResult,
    RunMetrics,
    RunRequest,
    RunResult,
    TaskInput,
    WorkflowOutput,
    utc_now,
)
from multi_agent_research.provenance import capture_run_provenance
from multi_agent_research.storage import FileRunStore
from multi_agent_research.workflows import Workflow


class ExperimentRunner:
    def __init__(
        self,
        *,
        llm: LLMClient,
        store: FileRunStore | None = None,
    ) -> None:
        self.llm = llm
        self.store = store
        self.provenance, self.source_snapshot = capture_run_provenance()

    async def run(
        self,
        *,
        task: TaskInput,
        workflow: Workflow,
        experiment_id: str,
        call_metadata: dict[str, Any] | None = None,
    ) -> RunResult:
        run_id = str(uuid4())
        request = RunRequest(
            id=run_id,
            experiment_id=experiment_id,
            task=task,
            workflow=workflow.spec(),
            provenance=self.provenance,
        )
        context = RunContext(
            run_id=run_id,
            task=task,
            workflow_name=workflow.name,
            llm=self.llm,
            call_metadata=call_metadata,
        )
        started_at = utc_now()
        started_clock = monotonic()
        final_answer: str | None = None
        output: WorkflowOutput | None = None
        inconclusive: InconclusiveResult | None = None
        error: CallError | None = None
        status = "success"

        try:
            raw_response = await workflow.run(task, context)
            output = WorkflowOutput.from_response(
                raw_response,
                task.answer_spec,
            )
            final_answer = output.answer
        except AggregationInconclusive as exc:
            status = "inconclusive"
            inconclusive = InconclusiveResult(
                type=type(exc).__name__,
                message=str(exc),
                details=exc.details,
            )
            context.emit(
                "run_inconclusive",
                inconclusive=inconclusive.model_dump(),
            )
        except Exception as exc:
            status = "failed"
            error = CallError(
                type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            )
            context.emit("run_failed", error=error.model_dump())

        ended_at = utc_now()
        wall_time_ms = (monotonic() - started_clock) * 1000
        calls = sorted(context.calls, key=lambda call: call.sequence)
        stage_answers = sorted(
            context.stage_answers,
            key=lambda stage_answer: stage_answer.sequence,
        )
        result = RunResult(
            run_id=run_id,
            experiment_id=experiment_id,
            task_id=task.id,
            workflow=workflow.spec(),
            provenance=self.provenance,
            status=status,
            final_answer=final_answer,
            output=output,
            inconclusive=inconclusive,
            error=error,
            started_at=started_at,
            ended_at=ended_at,
            metrics=RunMetrics.from_calls(calls, wall_time_ms),
            calls=calls,
            stage_answers=stage_answers,
            events=context.events,
        )
        if self.store:
            self.store.save(
                request,
                result,
                source_snapshot=self.source_snapshot,
            )
        return result
