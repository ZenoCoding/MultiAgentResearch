# Smoke Task: Unwinding Yo-Yo Acceleration

This mechanics task is a second end-to-end smoke question for every workflow.
It is closed-ended, text-only, exactly scoreable, and has a short reference
derivation that makes manual grading straightforward.

## Model-visible task

A yo-yo consists of two massive uniform disks of radius `R` connected by a
thin axle. A thick, negligible-mass string is wrapped many times around the
axle. Initially, the outermost layer of string is a distance `R` from the
axle. The end of the string is held fixed and the yo-yo is dropped from rest.
Assume energy losses are negligible and the string always remains vertical.

Between release and the moment the string completely unwinds, which statement
about the yo-yo's acceleration is true?

- A. It is always zero.
- B. It points downward, but decreases in magnitude over time.
- C. It points downward and has constant magnitude.
- D. It points downward, but increases in magnitude over time.
- E. None of the above.

## Reference data

- Task ID: `smoke-yoyo-unwinding-v1`
- Answer type: multiple choice
- Gold answer: `E`
- Domain: classical mechanics, rotational dynamics
- Tools required: none
- Source: transcribed from the user-provided question image

Keep this section outside the model-visible `TaskInput`.

## Verification

Let `z` be the distance fallen, `l` the string's total length, and `r` the
instantaneous radius of the remaining winding. For a uniformly thick string
wrapped around a thin axle, the remaining string volume gives

```text
r^2 / R^2 = (l - z) / l.
```

The instantaneous no-slip relation is `v = r*omega`, but because `r` changes,
its derivative is

```text
a = r*alpha + (dr/dt)*omega,
```

not `a = r*alpha`. The latter was the incorrect constant-radius assumption in
the original verification.

For total disk mass `M`, the two uniform disks have `I = M*R^2/2`.
Conservation of energy gives

```text
M*g*z = (1/2)*M*v^2 + (1/2)*I*omega^2.
```

Substituting `omega = v/r` and the expression for `r(z)` yields

```text
v^2 = 2*g*z / (1 + l/(2*(l - z)))
    = 4*g*z*(l - z) / (3*l - 2*z).
```

This speed starts at zero, reaches a maximum at
`z/l = (3 - sqrt(3))/2`, and returns to zero as `z` approaches `l`.
Because `a = (1/2)*d(v^2)/dz`, the acceleration is downward before that
maximum, zero at the maximum, and upward afterward. None of A through D
describes the entire descent, so the correct answer is `E`.

## Intended use

Use this as another smoke question, not as evidence for the main research
hypotheses. It checks whether every workflow runs, emits a valid
multiple-choice answer, records intermediate answers, and produces reasoning
that can be compared with the reference derivation.

Run all workflows with:

```bash
./examples/run-yoyo-smoke-workflows.sh <model>
```

For the matched GPT-5.4 nano comparison using `none` and `medium` reasoning,
run:

```bash
./examples/run-yoyo-reasoning-suites.sh
```

Each effort suite runs solo, three-agent sampling, six-agent sampling,
self-critique, three-agent debate with two rounds, and supervisor. Sampling
and debate use plurality voting, with a judge called only to break a top-vote
tie. Judge and supervisor reasoning effort match the primary-agent effort.

Run the separate adversarial-debate variant at both effort levels with:

```bash
./examples/run-yoyo-adversarial-debate.sh
```
