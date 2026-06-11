#!/usr/bin/env bash
set -euo pipefail

model="openai/gpt-5.4-nano"
task_id="smoke-yoyo-unwinding-v1"
output_root="results/yoyo-adversarial-debate"
prompt="A yo-yo consists of two massive uniform disks of radius R connected by a
thin axle. A thick, negligible-mass string is wrapped many times around the
axle. Initially, the outermost layer of string is a distance R from the axle.
The end of the string is held fixed and the yo-yo is dropped from rest. Assume
energy losses are negligible and the string always remains vertical.

Between release and the moment the string completely unwinds, which statement
about the yo-yo's acceleration is true?

A. It is always zero.
B. It points downward, but decreases in magnitude over time.
C. It points downward and has constant magnitude.
D. It points downward, but increases in magnitude over time.
E. None of the above."

run_debate() {
  local effort=$1

  echo
  echo "==> adversarial debate effort=$effort"
  uv run mar \
    --workflow adversarial-debate \
    --model "$model" \
    --experiment-id adversarial-debate-3x2 \
    --task-id "$task_id" \
    --output-dir "$output_root/$effort" \
    --prompt "$prompt" \
    --answer-type multiple_choice \
    --choice "A=It is always zero." \
    --choice "B=It points downward, but decreases in magnitude over time." \
    --choice "C=It points downward and has constant magnitude." \
    --choice "D=It points downward, but increases in magnitude over time." \
    --choice "E=None of the above." \
    --include-confidence \
    --reasoning-effort "$effort" \
    --judge-reasoning-effort "$effort" \
    --agents 3 \
    --rounds 2 \
    --aggregation plurality_vote \
    --vote-tie-break judge \
    --debate-peer-view full_response
}

run_debate none
run_debate medium
