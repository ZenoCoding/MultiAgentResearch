#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <model>" >&2
  exit 2
fi

model=$1
experiment_id="baseline-grid-parity"
task_id="synthetic-grid-parity-v1"
prompt='How many 6 by 6 matrices with entries in {0, 1} satisfy all three conditions?

1. Every row contains an even number of 1s.
2. Every column contains an even number of 1s.
3. All six rows are pairwise distinct.

Choose one:
A. 33,554,432
B. 20,389,320
C. 19,998,720
D. 652,458,240
E. 27,776'

common=(
  --model "$model"
  --experiment-id "$experiment_id"
  --task-id "$task_id"
  --prompt "$prompt"
  --answer-type multiple_choice
  --choice "A=33,554,432"
  --choice "B=20,389,320"
  --choice "C=19,998,720"
  --choice "D=652,458,240"
  --choice "E=27,776"
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
