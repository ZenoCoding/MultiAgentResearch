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

SHORT_ANSWER_SEMANTIC_VOTE_PROMPT = PromptTemplate(
    name="workflow.vote.short_answer_semantic",
    version="1.0.0",
    template=(
        "Perform $mode aggregation over $candidate_count short-answer "
        "candidates. Group final answers that are semantically equivalent even "
        "when wording, abbreviations, units, or formatting differ. For "
        "plurality_vote, select the largest semantic group. For majority_vote, "
        "select a group only if it contains more than half of all candidates; "
        "otherwise report no majority. Do not replace the vote winner with the "
        "answer you independently think is correct. If the largest groups are "
        "tied, apply this tie policy: $tie_policy. For inconclusive, report an "
        "inconclusive vote. For first, choose the tied group whose earliest "
        "candidate appears first. For judge, use candidate reasoning only to "
        "choose among the tied groups. Begin with exactly "
        "<vote_status>winner</vote_status> or "
        "<vote_status>inconclusive</vote_status>. For a winner, return one "
        "candidate's final answer from the winning group using the original "
        "task's required answer format.\n\n$candidates"
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

CROSS_EXAMINATION_CLAIM_PROMPT = PromptTemplate(
    name="workflow.cross_examination.claims",
    version="1.0.0",
    template=(
        "Extract the argumentative core of your proposed solution. Do not repeat "
        "the full solution and do not revise it yet. Return exactly two critical "
        "claims, the most important assumption, and the weakest step using this "
        "format:\n"
        "<claim id=\"C1\">...</claim>\n"
        "<claim id=\"C2\">...</claim>\n"
        "<assumption>...</assumption>\n"
        "<weakest_step>...</weakest_step>"
    ),
)

CROSS_EXAMINATION_CHALLENGE_PROMPT = PromptTemplate(
    name="workflow.cross_examination.challenge",
    version="1.1.0",
    template=(
        "Cross-examine $target_agent. Select one specific claim, assumption, or "
        "step below that could change the final answer. Even if you initially agree on the final answer, "
        "treat that agreement as a correlated-error warning. Be highly critical: actively search for "
        "hidden constancy assumptions, ignored terms (such as in kinematics or derivatives), or simplifications "
        "that could fail at the boundaries. Ask one precise question or give one concrete counterexample, "
        "derivation check, or boundary test. Do not restate your full solution. Address $target_agent directly "
        "and return only the challenge.\n\n"
        "$target_agent initial solution:\n$target_initial\n\n"
        "$target_agent claim dossier:\n$target_claims\n\n"
        "Prior cross-examination transcript:\n$prior_transcript"
    ),
)

CROSS_EXAMINATION_RESPONSE_PROMPT = PromptTemplate(
    name="workflow.cross_examination.response",
    version="1.1.0",
    template=(
        "$challenger_agent challenged you as follows:\n$challenge\n\n"
        "Answer that exact challenge directly. Defend your solution rigorously. Do not concede "
        "merely because the challenger presents a different model or formula; double-check your own "
        "fundamental conservation laws (e.g., energy, momentum) and boundary behavior first. Only "
        "concede if you identify an explicit, undeniable logical or mathematical error in your own "
        "derivation. Do not restate your full solution. Return only the response.\n\n"
        "Prior cross-examination transcript:\n$prior_transcript"
    ),
)

CROSS_EXAMINATION_VERDICT_PROMPT = PromptTemplate(
    name="workflow.cross_examination.verdict",
    version="1.1.0",
    template=(
        "You challenged $target_agent:\n$challenge\n\n"
        "$target_agent responded:\n$response\n\n"
        "Judge strictly whether the response fully and correctly resolved your challenge. Be highly "
        "skeptical: do not accept hand-waving, unverified simplifications, or unproven assumptions. "
        "Begin with exactly RESOLVED, UNRESOLVED, or CONCEDED, followed by one short sentence of justification."
    ),
)

CROSS_EXAMINATION_FINAL_REVISION_PROMPT = PromptTemplate(
    name="workflow.cross_examination.final_revision",
    version="1.1.0",
    template=(
        "Privately revise your solution after the cross-examination transcript "
        "below. Do not follow headcount and do not assume a challenge is resolved "
        "just because a peer claimed it was. Perform a ground-up independent re-derivation "
        "of any critical physics or mathematical steps yourself to verify them from first "
        "principles before deciding. Return your complete final solution.\n\n"
        "Your initial solution:\n$initial_response\n\n"
        "Your claim dossier:\n$claims\n\n"
        "Cross-examination transcript:\n$transcript"
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
