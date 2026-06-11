#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <model>" >&2
  exit 2
fi

model=$1
experiment_id="smoke-yoyo-unwinding"
task_id="smoke-yoyo-unwinding-v1"
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

common=(
  --model "$model"
  --experiment-id "$experiment_id"
  --task-id "$task_id"
  --prompt "$prompt"
  --answer-type multiple_choice
  --choice "A=It is always zero."
  --choice "B=It points downward, but decreases in magnitude over time."
  --choice "C=It points downward and has constant magnitude."
  --choice "D=It points downward, but increases in magnitude over time."
  --choice "E=None of the above."
  --include-confidence
)

run_workflow() {
  local workflow=$1
  shift
  echo
  echo "==> $workflow"
  uv run mar "${common[@]}" --workflow "$workflow" "$@"
}

run_workflow solo
run_workflow sample --agents 3 --aggregation plurality_vote
run_workflow self-critic --rounds 1
run_workflow debate \
  --agents 3 \
  --rounds 1 \
  --aggregation plurality_vote \
  --debate-peer-view full_response
run_workflow supervisor --rounds 1
