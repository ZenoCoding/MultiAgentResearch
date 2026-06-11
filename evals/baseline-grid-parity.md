# Baseline Task: Distinct Even-Parity Rows

This synthetic task is a quick end-to-end baseline for every workflow. It is
closed-ended, text-only, exactly scoreable, difficult enough to produce
plausible disagreements, and does not copy a benchmark item.

## Model-visible task

How many 6 by 6 matrices with entries in `{0, 1}` satisfy all three
conditions?

1. Every row contains an even number of `1`s.
2. Every column contains an even number of `1`s.
3. All six rows are pairwise distinct.

Choose one:

- A. `33,554,432`
- B. `20,389,320`
- C. `19,998,720`
- D. `652,458,240`
- E. `27,776`

## Reference data

- Task ID: `synthetic-grid-parity-v1`
- Answer type: multiple choice
- Gold answer: `C`
- Domain: combinatorics, finite vector spaces
- Tools required: none

Keep this section outside the model-visible `TaskInput`.

## Verification

Each valid row is an element of the five-dimensional even-parity subspace
`G` of `F_2^6`, so there are 32 possible rows. The column condition says that
the bitwise XOR of the six selected rows is zero.

First count unordered six-element subsets of `G` with XOR zero. Character
filtering gives

```text
1/32 * (C(32, 6) + 31 * [t^6](1 + t)^16(1 - t)^16)
```

Every nontrivial character is `+1` on 16 elements and `-1` on 16 elements.
Since

```text
[t^6](1 - t^2)^16 = -C(16, 3) = -560,
```

the number of unordered subsets is

```text
(C(32, 6) - 31*C(16, 3)) / 32 = 27,776.
```

The six distinct rows can be ordered in `6!` ways, giving

```text
27,776 * 6! = 19,998,720.
```

The distractors identify useful failure modes:

- A allows repeated rows.
- B incorrectly assumes subset XORs are uniformly distributed.
- D enforces row parity and distinctness but ignores column parity.
- E finds the correct unordered subsets but forgets row order.

## Intended use

Use this as a smoke baseline, not as evidence for the main research
hypotheses. A single synthetic problem cannot estimate accuracy or establish
whether one workflow is generally better. It can verify that every workflow
runs, emits valid multiple-choice output, records intermediate answers, and
produces qualitatively inspectable revisions.

Run all workflows with:

```bash
./examples/run-baseline-workflows.sh <model>
```
