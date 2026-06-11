from __future__ import annotations

from multi_agent_research.models import AgentSpec, PromptReference, PromptTemplate


JUDGE_SYSTEM_PROMPT = PromptTemplate(
    name="agent.judge.system",
    version="1.0.0",
    template="Act as a careful, impartial answer judge.",
)

SUPERVISOR_SYSTEM_PROMPT = PromptTemplate(
    name="agent.supervisor.system",
    version="1.0.0",
    template="Act as a strict supervisor who catches substantive errors.",
)

JUDGE_SELECTION_PROMPT = PromptTemplate(
    name="workflow.judge.selection",
    version="2.0.0",
    template=(
        "Choose or synthesize the best answer to the original task. "
        "Follow the task's required answer format.\n\n$candidates"
    ),
)

TIE_BREAK_JUDGE_PROMPT = PromptTemplate(
    name="workflow.judge.tie_break",
    version="1.0.0",
    template=(
        "The vote is tied among these final answers: $tied_answers. "
        "Choose the best answer to the original task using the tied candidates' "
        "reasoning. Your final answer must be one of the tied answers. "
        "Follow the task's required answer format.\n\n$candidates"
    ),
)

SELF_CRITIC_REVISION_PROMPT = PromptTemplate(
    name="workflow.self_critic.revision",
    version="1.0.0",
    template=(
        "Critically inspect your answer for factual, logical, and "
        "instruction-following errors. Return a revised final answer only."
    ),
)

DEBATE_REVIEW_PROMPT = PromptTemplate(
    name="workflow.debate.peer_review",
    version="1.0.0",
    template=(
        "Review the other agents' answers below. Correct your answer when they "
        "expose a real error, but do not defer merely because they disagree. "
        "Return your updated answer only.\n\nOther agents:\n$peer_answers"
    ),
)

SUPERVISOR_REVIEW_PROMPT = PromptTemplate(
    name="workflow.supervisor.review",
    version="2.0.0",
    template=(
        "Review the proposed answer to the task. Begin with APPROVE if it is "
        "ready, or REVISE followed by concrete feedback.\n\n"
        "Proposed answer:\n$answer"
    ),
)

WORKER_REVISION_PROMPT = PromptTemplate(
    name="workflow.supervisor.worker_revision",
    version="1.0.0",
    template=(
        "Revise your answer using this supervisor feedback. Return the revised "
        "final answer only.\n\n$feedback"
    ),
)


def system_prompt_template(agent: AgentSpec) -> PromptTemplate | None:
    if not agent.system_prompt:
        return None
    return PromptTemplate(
        name=agent.system_prompt_name or f"agent.{agent.id}.system",
        version=agent.system_prompt_version,
        template=agent.system_prompt,
    )


def system_prompt_reference(agent: AgentSpec) -> PromptReference | None:
    prompt = system_prompt_template(agent)
    return prompt.reference() if prompt else None


def unique_prompts(prompts: list[PromptTemplate | None]) -> list[PromptTemplate]:
    unique: dict[tuple[str, str, str], PromptTemplate] = {}
    for prompt in prompts:
        if prompt is None:
            continue
        key = (prompt.name, prompt.version, prompt.content_sha256)
        unique[key] = prompt
    return sorted(
        unique.values(),
        key=lambda prompt: (prompt.name, prompt.version, prompt.content_sha256),
    )
