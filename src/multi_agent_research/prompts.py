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

DEBATE_DERIVATION_ROLE_PROMPT = PromptTemplate(
    name="workflow.debate.role.derivation",
    version="1.0.0",
    template=(
        "Act as the first-principles solver. Derive the answer independently, "
        "state the assumptions that your derivation needs, and verify the "
        "critical transformations rather than relying on a familiar formula."
    ),
)

DEBATE_ASSUMPTION_AUDITOR_ROLE_PROMPT = PromptTemplate(
    name="workflow.debate.role.assumption_auditor",
    version="1.0.0",
    template=(
        "Act as the assumption auditor. Solve independently while looking for "
        "hidden constancy assumptions, quantities that may vary, invalid "
        "algebra or calculus steps, and conclusions that fail at boundaries "
        "or limiting cases."
    ),
)

DEBATE_ALTERNATIVE_METHOD_ROLE_PROMPT = PromptTemplate(
    name="workflow.debate.role.alternative_method",
    version="1.0.0",
    template=(
        "Act as the alternative-method verifier. Avoid the most obvious "
        "solution path when possible; use a second derivation, invariant, "
        "counterexample, dimensional check, or endpoint behavior to try to "
        "falsify the apparent answer."
    ),
)

DEBATE_ADVERSARIAL_CHALLENGE_PROMPT = PromptTemplate(
    name="workflow.debate.adversarial.challenge",
    version="1.0.0",
    template=(
        "Do not treat agreement or confidence as evidence. Identify the most "
        "consequential assumption or inference shared by these answers, build "
        "the strongest plausible case for a different conclusion, and test "
        "the disputed step using an independent method or limiting case. "
        "Revise only after that challenge. Return your updated answer only."
        "\n\nOther agents:\n$peer_answers"
    ),
)

DEBATE_ADVERSARIAL_UNANIMOUS_PROMPT = PromptTemplate(
    name="workflow.debate.adversarial.unanimous_challenge",
    version="1.0.0",
    template=(
        "All visible agents currently agree. Treat this unanimity as a "
        "correlated-error warning, not confirmation. Locate the shared "
        "assumption that would make every answer fail, construct and test at "
        "least one serious alternative conclusion, and re-derive the critical "
        "step without copying the shared method. Return your updated answer "
        "only.\n\nOther agents:\n$peer_answers"
    ),
)

DEBATE_ADVERSARIAL_RESOLUTION_PROMPT = PromptTemplate(
    name="workflow.debate.adversarial.resolution",
    version="1.0.0",
    template=(
        "Resolve the competing claims by independently checking their critical "
        "equations and assumptions. Give no weight to headcount. Prefer a "
        "claim only when its derivation, counterexample, or boundary behavior "
        "survives verification. Return your best updated answer only."
        "\n\nOther agents:\n$peer_answers"
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
