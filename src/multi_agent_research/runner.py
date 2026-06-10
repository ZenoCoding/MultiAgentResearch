from __future__ import annotations

from time import monotonic
from uuid import uuid4

from multi_agent_research.context import RunContext
from multi_agent_research.llm import LLMClient
from multi_agent_research.models import (
    CallError,
    RunMetrics,
    RunRequest,
    RunResult,
    TaskInput,
    WorkflowOutput,
    utc_now,
)
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

    async def run(
        self,
        *,
        task: TaskInput,
        workflow: Workflow,
        experiment_id: str,
    ) -> RunResult:
        run_id = str(uuid4())
        request = RunRequest(
            id=run_id,
            experiment_id=experiment_id,
            task=task,
            workflow=workflow.spec(),
        )
        context = RunContext(
            run_id=run_id,
            task=task,
            workflow_name=workflow.name,
            llm=self.llm,
        )
        started_at = utc_now()
        started_clock = monotonic()
        final_answer: str | None = None
        output: WorkflowOutput | None = None
        error: CallError | None = None
        status = "success"

        try:
            raw_response = await workflow.run(task, context)
            output = WorkflowOutput.from_response(
                raw_response,
                task.answer_spec,
            )
            final_answer = output.answer
        except Exception as exc:
            status = "failed"
            error = CallError(type=type(exc).__name__, message=str(exc))
            context.emit("run_failed", error=error.model_dump())

        ended_at = utc_now()
        wall_time_ms = (monotonic() - started_clock) * 1000
        calls = sorted(context.calls, key=lambda call: call.sequence)
        result = RunResult(
            run_id=run_id,
            experiment_id=experiment_id,
            task_id=task.id,
            workflow=workflow.spec(),
            status=status,
            final_answer=final_answer,
            output=output,
            error=error,
            started_at=started_at,
            ended_at=ended_at,
            metrics=RunMetrics.from_calls(calls, wall_time_ms),
            calls=calls,
            events=context.events,
        )
        if self.store:
            self.store.save(request, result)
        return result
