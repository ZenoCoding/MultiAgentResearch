from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from multi_agent_research.aggregation import VotingConfig
from multi_agent_research.litellm_client import LiteLLMClient
from multi_agent_research.models import (
    AgentSpec,
    AnswerChoice,
    AnswerSpec,
    TaskInput,
)
from multi_agent_research.prompt_overrides import load_prompt_overrides, overridden
from multi_agent_research.prompts import (
    DEBATE_REVIEW_PROMPT,
    JUDGE_SELECTION_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    SELF_CRITIC_REVISION_PROMPT,
    SUPERVISOR_REVIEW_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
    WORKER_REVISION_PROMPT,
)
from multi_agent_research.runner import ExperimentRunner
from multi_agent_research.storage import FileRunStore
from multi_agent_research.workflows import (
    DebateWorkflow,
    IndependentSampleWorkflow,
    SelfCriticWorkflow,
    SoloWorkflow,
    SupervisorWorkflow,
    Workflow,
)


def main() -> None:
    load_dotenv()
    args = _parser().parse_args()
    result = asyncio.run(_run(args))
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "status": result.status,
                "final_answer": result.final_answer,
                "output": result.output.model_dump() if result.output else None,
                "workflow": result.workflow.model_dump(),
                "metrics": result.metrics.model_dump(),
            },
            indent=2,
        )
    )
    if result.status == "failed":
        raise SystemExit(1)


async def _run(args: argparse.Namespace):
    parameters = _model_parameters(args)
    prompt_overrides = load_prompt_overrides(args.prompt_overrides)
    base_agent = AgentSpec(
        id="agent-1",
        model=args.model,
        system_prompt=args.system_prompt,
        system_prompt_name="agent.primary.system",
        system_prompt_version=args.system_prompt_version,
        reasoning_effort=args.reasoning_effort,
        service_tier=args.service_tier,
        parameters=parameters,
    )
    workflow = _workflow(args, base_agent, parameters, prompt_overrides)
    task = TaskInput.from_prompt(
        id=args.task_id or str(uuid4()),
        prompt=args.prompt,
        answer_spec=AnswerSpec(
            type=args.answer_type,
            choices=[_parse_choice(choice) for choice in args.choice],
            include_explanation=not args.no_explanation,
            include_confidence=args.include_confidence,
        ),
    )
    runner = ExperimentRunner(
        llm=LiteLLMClient(),
        store=FileRunStore(Path(args.output_dir)),
    )
    return await runner.run(
        task=task,
        workflow=workflow,
        experiment_id=args.experiment_id,
    )


def _workflow(
    args: argparse.Namespace,
    base_agent: AgentSpec,
    parameters: dict,
    prompt_overrides: dict,
) -> Workflow:
    if args.workflow == "solo":
        return SoloWorkflow(base_agent)
    if args.workflow == "self-critic":
        return SelfCriticWorkflow(
            base_agent,
            rounds=args.rounds,
            revision_prompt=overridden(
                SELF_CRITIC_REVISION_PROMPT,
                prompt_overrides,
            ),
        )

    agents = [
        AgentSpec(
            id=f"agent-{index + 1}",
            model=args.model,
            system_prompt=args.system_prompt,
            system_prompt_name="agent.primary.system",
            system_prompt_version=args.system_prompt_version,
            reasoning_effort=args.reasoning_effort,
            service_tier=args.service_tier,
            parameters=parameters,
        )
        for index in range(args.agents)
    ]
    judge = None
    if args.aggregation == "judge":
        judge_system = overridden(
            JUDGE_SYSTEM_PROMPT,
            prompt_overrides,
        )
        judge = AgentSpec(
            id="judge",
            model=args.judge_model or args.model,
            system_prompt=judge_system.template,
            system_prompt_name=judge_system.name,
            system_prompt_version=judge_system.version,
            reasoning_effort=args.judge_reasoning_effort or args.reasoning_effort,
            service_tier=args.judge_service_tier or args.service_tier,
            parameters=parameters,
        )
    voting = VotingConfig(
        tie_break=args.vote_tie_break,
        random_seed=args.vote_seed,
        invalid_ballot_policy=args.invalid_ballot_policy,
    )
    if args.workflow == "sample":
        return IndependentSampleWorkflow(
            agents,
            judge,
            judge_prompt=overridden(
                JUDGE_SELECTION_PROMPT,
                prompt_overrides,
            ),
            parallel=not args.sequential,
            aggregation=args.aggregation,
            voting=voting,
        )
    if args.workflow == "debate":
        return DebateWorkflow(
            agents,
            judge,
            rounds=args.rounds,
            debate_prompt=overridden(
                DEBATE_REVIEW_PROMPT,
                prompt_overrides,
            ),
            judge_prompt=overridden(
                JUDGE_SELECTION_PROMPT,
                prompt_overrides,
            ),
            parallel=not args.sequential,
            aggregation=args.aggregation,
            voting=voting,
        )
    if args.workflow == "supervisor":
        supervisor_system = overridden(
            SUPERVISOR_SYSTEM_PROMPT,
            prompt_overrides,
        )
        supervisor = AgentSpec(
            id="supervisor",
            model=args.judge_model or args.model,
            system_prompt=supervisor_system.template,
            system_prompt_name=supervisor_system.name,
            system_prompt_version=supervisor_system.version,
            reasoning_effort=(
                args.supervisor_reasoning_effort
                or args.judge_reasoning_effort
                or args.reasoning_effort
            ),
            service_tier=(
                args.supervisor_service_tier
                or args.judge_service_tier
                or args.service_tier
            ),
            parameters=parameters,
        )
        return SupervisorWorkflow(
            worker=base_agent,
            supervisor=supervisor,
            max_revisions=args.rounds,
            review_prompt=overridden(
                SUPERVISOR_REVIEW_PROMPT,
                prompt_overrides,
            ),
            revision_prompt=overridden(
                WORKER_REVISION_PROMPT,
                prompt_overrides,
            ),
        )
    raise ValueError(f"Unsupported workflow: {args.workflow}")


def _model_parameters(args: argparse.Namespace) -> dict:
    parameters: dict = {}
    if args.temperature is not None:
        parameters["temperature"] = args.temperature
    if args.max_tokens is not None:
        parameters["max_tokens"] = args.max_tokens
    return parameters


def _parse_choice(value: str) -> AnswerChoice:
    label, separator, text = value.partition("=")
    return AnswerChoice(
        label=label.strip(),
        text=text.strip() if separator else None,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mar",
        description="Run a standardized LLM agent workflow.",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--answer-type",
        choices=[
            "free_text",
            "short_answer",
            "multiple_choice",
            "number",
            "json",
            "code",
        ],
        default="short_answer",
    )
    parser.add_argument(
        "--choice",
        action="append",
        default=[],
        help="Answer choice as LABEL or LABEL=text; repeat for each choice.",
    )
    parser.add_argument("--include-confidence", action="store_true")
    parser.add_argument("--no-explanation", action="store_true")
    parser.add_argument(
        "--workflow",
        choices=["solo", "sample", "self-critic", "debate", "supervisor"],
        default="solo",
    )
    parser.add_argument("--experiment-id", default="manual")
    parser.add_argument("--task-id")
    parser.add_argument("--judge-model")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--system-prompt-version", default="inline")
    parser.add_argument("--prompt-overrides")
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--reasoning-effort",
        help="Reasoning effort for primary agents, for example low or high.",
    )
    parser.add_argument(
        "--judge-reasoning-effort",
        help="Override reasoning effort for the judge.",
    )
    parser.add_argument(
        "--supervisor-reasoning-effort",
        help="Override reasoning effort for the supervisor.",
    )
    parser.add_argument(
        "--service-tier",
        choices=["auto", "default", "flex", "priority"],
        help="Processing tier for primary agents.",
    )
    parser.add_argument(
        "--judge-service-tier",
        choices=["auto", "default", "flex", "priority"],
        help="Override processing tier for the judge.",
    )
    parser.add_argument(
        "--supervisor-service-tier",
        choices=["auto", "default", "flex", "priority"],
        help="Override processing tier for the supervisor.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable parallel calls in independent workflow phases.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["judge", "majority_vote", "plurality_vote"],
        default="judge",
        help="How sample and debate workflows select their final answer.",
    )
    parser.add_argument(
        "--vote-tie-break",
        choices=["error", "first", "random"],
        default="error",
    )
    parser.add_argument("--vote-seed", type=int, default=0)
    parser.add_argument(
        "--invalid-ballot-policy",
        choices=["exclude", "error"],
        default="exclude",
    )
    parser.add_argument("--output-dir", default="results")
    return parser


if __name__ == "__main__":
    main()
