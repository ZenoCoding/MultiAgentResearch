#!/usr/bin/env bash
set -uo pipefail

model="openai/gpt-5.4-nano"
task_id="smoke-yoyo-unwinding-v1"
output_root="results/yoyo-reasoning-comparison"
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

failures=()

run_variant() {
  local effort=$1
  local variant=$2
  local workflow=$3
  shift 3

  echo
  echo "==> effort=$effort variant=$variant workflow=$workflow"

  if ! uv run mar \
    --model "$model" \
    --experiment-id "$variant" \
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
    --supervisor-reasoning-effort "$effort" \
    --workflow "$workflow" \
    "$@"
  then
    failures+=("$effort/$variant")
  fi
}

run_suite() {
  local effort=$1

  run_variant "$effort" solo solo
  run_variant "$effort" sample-3 sample \
    --agents 3 \
    --aggregation plurality_vote \
    --vote-tie-break judge
  run_variant "$effort" sample-6 sample \
    --agents 6 \
    --aggregation plurality_vote \
    --vote-tie-break judge
  run_variant "$effort" self-critic self-critic --rounds 1
  run_variant "$effort" debate-3x2 debate \
    --agents 3 \
    --rounds 2 \
    --aggregation plurality_vote \
    --vote-tie-break judge \
    --debate-peer-view full_response
  run_variant "$effort" supervisor supervisor --rounds 1
}

run_suite none
run_suite medium

if (( ${#failures[@]} > 0 )); then
  printf '\nFailed variants:\n' >&2
  printf '  %s\n' "${failures[@]}" >&2
  exit 1
fi
