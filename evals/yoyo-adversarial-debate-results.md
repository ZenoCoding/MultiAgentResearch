# Yo-Yo Adversarial Debate Comparison

Gold answer: **E**

Model: `openai/gpt-5.4-nano`

Both workflows used three agents, two debate rounds, full peer responses,
plurality voting, and a judge available only to break a tie. No tie occurred,
so neither workflow called the judge.

## Results

| Effort | Workflow | Initial answers | Round 1 | Round 2 | Final | Tokens | Cost |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| None | Standard debate | C, C, C | C, C, C | C, C, C | C | 11,658 | $0.00554 |
| None | Adversarial debate | C, C, B | B, B, B | B, B, B | B | 21,276 | $0.01072 |
| Medium | Standard debate | B, B, B | B, B, B | B, B, B | B | 28,649 | $0.02916 |
| Medium | Adversarial debate | C, E, E | E, E, E | E, E, E | E | 50,249 | $0.04665 |

Standard run IDs:

- None: `84fe2a3f-96d9-4fa7-809b-9ec9497db3ee`
- Medium: `dd1fce79-8421-415a-b72d-8a4aa45c1916`

Adversarial run IDs:

- None: `b5c2b9a7-9540-423a-88bb-08b2e8fc61b7`
- Medium: `c76630fa-3b8e-4787-8cee-8d7368421132`

## Interpretation

The standard medium debate had no recovery path because every initial agent
used the same invalid variable-radius derivation and selected B.

The adversarial medium roles changed the initial hypothesis distribution:

- The first-principles role selected C.
- The assumption-auditor role selected E.
- The alternative-method role selected E.

The first adversarial round then made all three agents verify the endpoint
energy behavior and converge to E. The second evidence-resolution round
retained E unanimously.

This supports the mechanism targeted by the new workflow: create materially
different initial approaches before peer influence, then verify claims rather
than treating agreement as evidence. It does not yet establish a general
accuracy gain. This is one task and one stochastic trial per condition, and
the successful medium adversarial run used about 75% more tokens than the
standard medium debate.

The no-reasoning adversarial run remained wrong. Its roles created initial
disagreement, but all agents converged on B after repeating the same invalid
step, `a = r*alpha`, without differentiating the changing-radius relation
`v = r*omega`. The prompts cannot replace reasoning capacity when no agent
successfully performs the decisive check.
