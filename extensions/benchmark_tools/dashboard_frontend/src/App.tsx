import { useState, useEffect, useMemo } from 'react';

// Interfaces for MART dashboard
interface ExperimentSummary {
  experiment_id: string;
  purpose: string;
  status: 'running' | 'complete' | 'partially graded' | 'incomplete' | 'failed';
  model: string;
  judge_model: string;
  grader_model: string;
  task_set: string;
  task_set_sha256: string;
  task_count: number;
  conditions_count: number;
  repetitions: number;
  expected_jobs: number;
  completed_jobs: number;
  graded_jobs: number;
  workflow_cost: number;
  grading_cost: number;
  total_cost: number;
  total_tokens: number;
  created_at: string;
  updated_at: string;
  best_condition: string | null;
  best_accuracy: number | null;
  best_ci_lower: number | null;
  best_ci_upper: number | null;
}

interface ConditionMetrics {
  condition: string;
  workflow: string;
  expected_jobs: number;
  completed_answer_jobs: number;
  graded_jobs: number;
  grading_failures: number;
  inconclusive_jobs: number;
  provider_execution_failures: number;
  contract_invalid_outputs: number;
  missing_jobs: number;
  attempts: number;
  retried_jobs: number;
  coverage_rate: number;
  planned_job_accuracy: number;
  planned_job_accuracy_ci_lower: number | null;
  planned_job_accuracy_ci_upper: number | null;
  valid_completed_accuracy: number | null;
  graded_accuracy: number | null;
  graded_accuracy_ci_lower: number | null;
  graded_accuracy_ci_upper: number | null;
  correct_valid_completed: number;
  total_tokens: number;
  input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  cost_usd: number;
  correct_per_dollar: number | null;
  correct_per_million_tokens: number | null;
  avg_wall_time_ms: number;
  calibration: {
    brier_score: number;
    expected_calibration_error: number;
    bins: Array<{
      bin_start: number;
      bin_end: number;
      count: number;
      avg_confidence: number;
      accuracy: number;
    }>;
  } | null;
}

interface PairedComparison {
  condition_a: string;
  condition_b: string;
  matched_completed_pairs: number;
  accuracy_a: number | null;
  accuracy_b: number | null;
  accuracy_delta_b_minus_a: number | null;
  accuracy_delta_ci_lower: number | null;
  accuracy_delta_ci_upper: number | null;
}

interface RunRow {
  experiment_id: string;
  job_id: string | null;
  job_key: string;
  condition: string;
  workflow: string;
  run_id: string;
  task_id: string;
  repetition: number;
  category: string;
  answer_type: string;
  status: string;
  outcome: string;
  attempt_count: number;
  successful_attempt_count: number;
  selected_attempt_ended_at: string;
  final_answer: string;
  expected_answer: string;
  correct: boolean | null;
  grading_status: string;
  grader_extracted_answer: string;
  grader_reasoning: string;
  grader_confidence: number | null;
  contract_valid: number;
  error_reason: string;
  model_calls: number;
  input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  total_tokens: number;
  cost_usd: number;
  wall_time_ms: number;
}

interface StageRow {
  condition: string;
  job_key: string;
  repetition: number;
  workflow: string;
  run_id: string;
  task_id: string;
  sequence: number;
  step: string;
  kind: string;
  agent_id: string;
  answer: string;
  correct: boolean | null;
  grading_status: string;
  grader_extracted_answer: string;
  grader_reasoning: string;
  grader_confidence: number | null;
  contract_valid: number;
  confidence: number | null;
}

interface TaskDetail {
  id: string;
  prompt: string;
  choices: Array<{ label: string; text: string }>;
  answer: string;
  answer_type: string;
  category: string;
}

interface ExperimentDetails {
  manifest: any;
  ledger: any;
  summary: {
    conditions: ConditionMetrics[];
    paired_comparisons: PairedComparison[];
    error_reasons: Record<string, number>;
    revision_transitions: Array<{ condition: string; transition: string; count: number }>;
    question_heatmap: Array<{
      task_id: string;
      repetition: number;
      condition: string;
      correct: boolean | null;
      outcome: string;
      category: string;
    }>;
  };
  runs: RunRow[];
  stages: StageRow[];
  tasks: TaskDetail[];
  warnings: string[];
}

interface RunDetailTrace {
  summary: any;
  workflow: any;
  task: any;
  agents: any[];
  phase_order: Array<{ id: string; label: string }>;
  stage_cards: any[];
  exchanges: any[];
  calls: any[];
  events: any[];
  output: any;
  error: any;
  inconclusive: any;
  provenance: any;
}
type ConditionIdentity = {
  workflow: string;
  effort: string | null;
  workerEffort: string | null;
  supervisorEffort: string | null;
  agents: number | null;
  rounds: number | null;
};

const WORKFLOW_ORDER = ['solo', 'sample', 'self-critic', 'debate', 'supervisor'];

const normalizeWorkflow = (workflow?: string, condition = '') => {
  const normalized = (workflow || '').replace(/_/g, '-').toLowerCase();
  if (normalized === 'independent-sample' || normalized === 'sampling') return 'sample';
  if (normalized === 'supervisor-worker') return 'supervisor';
  if (normalized && normalized !== 'unknown') return normalized;
  if (condition.startsWith('supervisor')) return 'supervisor';
  if (condition.startsWith('self-critic')) return 'self-critic';
  if (condition.startsWith('sample')) return 'sample';
  if (condition.startsWith('debate')) return 'debate';
  if (condition.startsWith('solo')) return 'solo';
  return 'unknown';
};

const getConditionIdentity = (condition: string, workflow?: string): ConditionIdentity => {
  const normalizedWorkflow = normalizeWorkflow(workflow, condition);
  const effortMatch = condition.match(/-e-([a-z0-9]+)/);
  const workerMatch = condition.match(/-(?:w|we)-([a-z0-9]+)/);
  const supervisorMatch = condition.match(/-(?:s|se)-([a-z0-9]+)/);
  const agentsMatch = condition.match(/-a(\d+)/);
  const roundsMatch = condition.match(/-r(\d+)/);

  return {
    workflow: normalizedWorkflow,
    effort: effortMatch?.[1] || null,
    workerEffort: workerMatch?.[1] || null,
    supervisorEffort: supervisorMatch?.[1] || null,
    agents: agentsMatch ? Number(agentsMatch[1]) : null,
    rounds: roundsMatch ? Number(roundsMatch[1]) : null,
  };
};

const formatWorkflowName = (workflow: string) => {
  if (workflow === 'sample') return 'Sampling';
  if (workflow === 'self-critic') return 'Self-critic';
  if (workflow === 'supervisor') return 'Supervisor';
  if (workflow === 'debate') return 'Debate';
  if (workflow === 'solo') return 'Solo';
  return workflow.replace(/-/g, ' ');
};

// Filters operate on stable workflow families. Lines use narrower comparable
// series so only one scale variable changes along a connected path.
const getGroupName = (c: { workflow?: string; condition: string }) =>
  getConditionIdentity(c.condition, c.workflow).workflow;

const getSeriesKey = (c: { workflow?: string; condition: string }) => {
  const identity = getConditionIdentity(c.condition, c.workflow);
  if (identity.workflow === 'solo') return 'solo|effort';
  if (identity.workflow === 'sample' || identity.workflow === 'self-critic') {
    return `${identity.workflow}|effort:${identity.effort || 'default'}`;
  }
  if (identity.workflow === 'debate') {
    return `debate|effort:${identity.effort || 'default'}|rounds:${identity.rounds ?? 'default'}`;
  }
  if (identity.workflow === 'supervisor') {
    return `supervisor|worker:${identity.workerEffort || 'default'}|supervisor:${identity.supervisorEffort || 'default'}`;
  }
  return `${identity.workflow}|default`;
};

const formatConditionConfig = (c: { workflow?: string; condition: string }) => {
  const identity = getConditionIdentity(c.condition, c.workflow);
  const effort = identity.effort || 'default';
  if (identity.workflow === 'solo') return `${effort} effort`;
  if (identity.workflow === 'sample') {
    return `${effort} effort · ${identity.agents ?? '?'} agents`;
  }
  if (identity.workflow === 'self-critic') {
    return `${effort} effort · ${identity.rounds ?? '?'} revision rounds`;
  }
  if (identity.workflow === 'debate') {
    return `${effort} effort · ${identity.agents ?? '?'} agents · ${identity.rounds ?? '?'} rounds`;
  }
  if (identity.workflow === 'supervisor') {
    return `worker ${identity.workerEffort || 'default'} · supervisor ${identity.supervisorEffort || 'default'} · max ${identity.rounds ?? '?'} rounds`;
  }
  return c.condition;
};

const formatConditionTag = (c: { workflow?: string; condition: string }) => {
  const identity = getConditionIdentity(c.condition, c.workflow);
  const workflow = formatWorkflowName(identity.workflow);
  if (identity.workflow === 'solo') return `${workflow} · ${identity.effort || 'default'}`;
  if (identity.workflow === 'sample') {
    return `${workflow} · ${identity.effort || 'default'} · a${identity.agents ?? '?'}`;
  }
  if (identity.workflow === 'self-critic') {
    return `${workflow} · ${identity.effort || 'default'} · r${identity.rounds ?? '?'}`;
  }
  if (identity.workflow === 'debate') {
    return `${workflow} · ${identity.effort || 'default'} · a${identity.agents ?? '?'}/r${identity.rounds ?? '?'}`;
  }
  if (identity.workflow === 'supervisor') {
    return `${workflow} · W ${identity.workerEffort || '?'} / S ${identity.supervisorEffort || '?'} · r${identity.rounds ?? '?'}`;
  }
  return c.condition;
};

const parseConditionName = (id: string) => {
  const identity = getConditionIdentity(id);
  return {
    name: identity.workflow,
    effort: formatConditionConfig({ condition: id, workflow: identity.workflow }),
  };
};

const getWorkflowIcon = (name: string) => {
  const n = name.toLowerCase();
  if (n.includes('solo')) {
    return (
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
        <rect x="3" y="11" width="18" height="10" rx="2" />
        <circle cx="12" cy="5" r="1.5" />
        <path d="M12 6.5v4.5M8 15.5h.01M16 15.5h.01" />
      </svg>
    );
  }
  if (n.includes('debate')) {
    return (
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      </svg>
    );
  }
  if (n.includes('critic')) {
    return (
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
        <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
        <path d="M9 12l2 2 4-4" />
      </svg>
    );
  }
  if (n.includes('supervisor')) {
    return (
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4M12 8h.01" />
    </svg>
  );
};

const getWorkflowColor = (name: string) => {
  const n = name.toLowerCase();
  if (n.includes('solo')) return '#3b82f6'; // Blue
  if (n.includes('debate')) return '#8b5cf6'; // Purple
  if (n.includes('critic')) return '#10b981'; // Emerald Green
  if (n.includes('supervisor')) return '#f97316'; // Orange
  return '#6b7280'; // Gray
};

const formatAvgCost = (val: number) => {
  if (val === 0) return '$0.00';
  if (val >= 1.0) return `$${val.toFixed(2)}`;
  if (val >= 0.01) return `$${val.toFixed(3)}`;
  return `$${val.toFixed(4)}`;
};

const formatAvgTime = (ms: number) => {
  if (ms <= 0) return '0s';
  const sec = ms / 1000;
  if (sec < 60) return `${Math.round(sec)}s`;
  const min = sec / 60;
  return `${Math.round(min)}m`;
};

const formatAvgTokens = (val: number) => {
  if (val <= 0) return '0';
  if (val >= 1000) return `${(val / 1000).toFixed(0)}k`;
  return Math.round(val).toString();
};

export default function App() {
  const [view, setView] = useState<'list' | 'detail' | 'compare'>('list');
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedExpId, setSelectedExpId] = useState<string | null>(null);
  const [details, setDetails] = useState<ExperimentDetails | null>(null);
  const [selectedRunsToCompare, setSelectedRunsToCompare] = useState<string[]>([]);
  const [comparisonResult, setComparisonResult] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Filters for experiment detail view
  const [activeTab, setActiveTab] = useState<'protocol' | 'health' | 'results' | 'efficiency' | 'per_question' | 'multi_agent' | 'calibration' | 'runs'>('results');
  const [filterSubject, setFilterSubject] = useState<string>('all');
  const [filterAnswerType, setFilterAnswerType] = useState<string>('all');
  const [baselineCondition, setBaselineCondition] = useState<string>('solo-e-low');
  const [efficiencyMetric, setEfficiencyMetric] = useState<'cost' | 'time' | 'tokens'>('cost');
  const [hoveredCondition, setHoveredCondition] = useState<string | null>(null);
  const [selectedGroups, setSelectedGroups] = useState<string[]>([]);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  // Deepswe dashboard states
  const [effortFilter, setEffortFilter] = useState<'best' | 'all'>('best');
  const [selectedWorkflows, setSelectedWorkflows] = useState<string[]>([]);
  const [showWorkflowDropdown, setShowWorkflowDropdown] = useState(false);

  // Search/Filter for runs table
  const [runSearchQuery, setRunSearchQuery] = useState('');

  // Selected cell detail run state
  const [selectedCellRunId, setSelectedCellRunId] = useState<string | null>(null);
  const [selectedCellTrace, setSelectedCellTrace] = useState<RunDetailTrace | null>(null);
  const [selectedCellMetadata, setSelectedCellMetadata] = useState<RunRow | null>(null);
  const [traceLoading, setTraceLoading] = useState(false);

  // Fetch experiments list on load
  useEffect(() => {
    fetchExperiments();
    const interval = setInterval(fetchExperiments, 10000); // refresh list every 10s
    return () => clearInterval(interval);
  }, []);

  const fetchExperiments = async () => {
    try {
      const res = await fetch('/api/experiments');
      if (!res.ok) throw new Error('Failed to fetch experiments');
      const data = await res.json();
      setExperiments(data);
    } catch (err: any) {
      console.error(err);
    }
  };

  // Poll details if experiment is active/running
  useEffect(() => {
    let interval: any;
    if (view === 'detail' && selectedExpId && details) {
      const summary = experiments.find(e => e.experiment_id === selectedExpId);
      if (summary && (summary.status === 'running' || summary.status === 'incomplete')) {
        interval = setInterval(() => {
          fetchDetails(selectedExpId, true);
        }, 5000); // poll active every 5s
      }
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [view, selectedExpId, details, experiments]);

  const fetchDetails = async (expId: string, isPoll = false) => {
    if (!isPoll) {
      setLoading(true);
      setErrorMsg(null);
      setDetails(null);
    }
    try {
      const res = await fetch(`/api/experiments/${expId}`);
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to load details');
      }
      const data = await res.json();
      setDetails(data);
      // Auto-set baseline condition based on discovered conditions
      const conditions = data.summary.conditions.map((c: any) => c.condition);
      if (conditions.length > 0) {
        const solo = conditions.find((c: string) => c.toLowerCase().includes('solo'));
        setBaselineCondition(solo || conditions[0]);
        if (!isPoll) {
          const wfs = (Array.from(new Set(data.summary.conditions.map((c: any) => getGroupName(c)))) as string[])
            .sort((a, b) => {
              const aIndex = WORKFLOW_ORDER.indexOf(a);
              const bIndex = WORKFLOW_ORDER.indexOf(b);
              return (aIndex === -1 ? 99 : aIndex) - (bIndex === -1 ? 99 : bIndex);
            });
          setSelectedGroups(wfs);

          const uniqueWfs = Array.from(
            new Set(
              data.summary.conditions.map((c: any) => parseConditionName(c.condition).name)
            )
          ) as string[];
          setSelectedWorkflows(uniqueWfs);
        }
      }
    } catch (err: any) {
      setErrorMsg(err.message);
    } finally {
      if (!isPoll) setLoading(false);
    }
  };

  const handleOpenDetail = (expId: string) => {
    setSelectedExpId(expId);
    setView('detail');
    setActiveTab('results');
    fetchDetails(expId);
  };

  const handleToggleSelectCompare = (expId: string) => {
    if (selectedRunsToCompare.includes(expId)) {
      setSelectedRunsToCompare(selectedRunsToCompare.filter(id => id !== expId));
    } else {
      setSelectedRunsToCompare([...selectedRunsToCompare, expId]);
    }
  };

  const handleCompare = async () => {
    if (selectedRunsToCompare.length < 2) return;
    setLoading(true);
    setErrorMsg(null);
    setView('compare');
    try {
      const res = await fetch('/api/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiment_ids: selectedRunsToCompare }),
      });
      if (!res.ok) throw new Error('Comparability check failed');
      const data = await res.json();
      setComparisonResult(data);
    } catch (err: any) {
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleOpenRunCellDetail = async (runRow: RunRow) => {
    if (!selectedExpId) return;
    setSelectedCellMetadata(runRow);
    setSelectedCellRunId(runRow.run_id);
    setSelectedCellTrace(null);
    setTraceLoading(true);
    try {
      const res = await fetch(`/api/experiments/${selectedExpId}/runs/${runRow.run_id}`);
      if (!res.ok) throw new Error('Failed to load run details');
      const data = await res.json();
      setSelectedCellTrace(data);
    } catch (err) {
      console.error(err);
    } finally {
      setTraceLoading(false);
    }
  };

  // Helper to format currency
  const formatUSD = (val: number) => `$${val.toFixed(4)}`;

  // Helper to format timestamps
  const formatTime = (isoStr: string) => {
    if (!isoStr) return 'n/a';
    try {
      const d = new Date(isoStr);
      return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      return isoStr;
    }
  };

  // Unique subjects/categories in current experiment
  const getSubjectsList = () => {
    if (!details) return [];
    return Array.from(new Set(details.tasks.map(t => t.category)));
  };

  // Filter logic for runs/tasks
  const getFilteredRuns = () => {
    if (!details) return [];
    return details.runs.filter(run => {
      const matchesSubject = filterSubject === 'all' || run.category === filterSubject;
      const matchesType = filterAnswerType === 'all' || run.answer_type === filterAnswerType;
      
      const rowText = `${run.task_id} ${run.condition} ${run.outcome} ${run.final_answer} ${run.error_reason}`.toLowerCase();
      const matchesSearch = runSearchQuery === '' || rowText.includes(runSearchQuery.toLowerCase());
      
      return matchesSubject && matchesType && matchesSearch;
    });
  };

  // Calculate filtered condition accuracies in frontend to respect dynamic filters
  const getFilteredAccuracies = () => {
    if (!details) return [];
    // Matched task IDs
    // Matched task IDs
    const matchedTaskIds = new Set(
      details.tasks
        .filter(t => (filterSubject === 'all' || t.category === filterSubject) && (filterAnswerType === 'all' || t.answer_type === filterAnswerType))
        .map(t => t.id)
    );

    return details.summary.conditions.map(cond => {
      // Find runs for this condition in matched tasks
      const condRuns = details.runs.filter(r => r.condition === cond.condition && matchedTaskIds.has(r.task_id));
      const expected = condRuns.length;
      const correct = condRuns.filter(r => r.correct === true).length;
      const graded = condRuns.filter(r => r.correct !== null).length;
      const valid = condRuns.filter(r => r.outcome === 'completed_valid').length;
      
      // Calculate dynamic Wilson intervals
      const planned_ci = calculateWilson(correct, expected);
      const graded_ci = calculateWilson(condRuns.filter(r => r.outcome === 'completed_valid' && r.correct === true).length, graded);

      return {
        condition: cond.condition,
        workflow: cond.workflow,
        expected,
        correct,
        graded,
        valid,
        planned_accuracy: expected > 0 ? correct / expected : 0,
        planned_ci_lower: planned_ci.lower,
        planned_ci_upper: planned_ci.upper,
        graded_accuracy: graded > 0 ? condRuns.filter(r => r.correct === true).length / graded : 0,
        graded_ci_lower: graded_ci.lower,
        graded_ci_upper: graded_ci.upper,
        cost: condRuns.reduce((sum, r) => sum + r.cost_usd, 0),
        tokens: condRuns.reduce((sum, r) => sum + r.total_tokens, 0),
        input_tokens: condRuns.reduce((sum, r) => sum + r.input_tokens, 0),
        output_tokens: condRuns.reduce((sum, r) => sum + r.output_tokens, 0),
        reasoning_tokens: condRuns.reduce((sum, r) => sum + r.reasoning_tokens, 0),
        avg_time: condRuns.length > 0 ? condRuns.reduce((sum, r) => sum + r.wall_time_ms, 0) / condRuns.length : 0,
      };
    });
  };

  // Wilson score interval formula in Javascript for reactive charts
  const calculateWilson = (successes: number, trials: number) => {
    if (trials <= 0) return { lower: 0, upper: 0 };
    const z = 1.95996; // 95%
    const p = successes / trials;
    const denominator = 1 + (z * z) / trials;
    const center = (p + (z * z) / (2 * trials)) / denominator;
    const spread = (z * Math.sqrt((p * (1 - p)) / trials + (z * z) / (4 * trials * trials))) / denominator;
    return {
      lower: Math.max(0, center - spread),
      upper: Math.min(1, center + spread),
    };
  };

  const allWorkflows = useMemo(() => {
    if (!details) return [];
    return Array.from(
      new Set(
        details.summary.conditions.map((c: any) => parseConditionName(c.condition).name)
      )
    ) as string[];
  }, [details]);

  const displayedAccuracies = useMemo(() => {
    const allData = getFilteredAccuracies();
    // Filter by selected workflows
    const filteredByWorkflow = allData.filter(c => {
      const parsed = parseConditionName(c.condition);
      if (selectedWorkflows.length === 0) return true;
      return selectedWorkflows.includes(parsed.name);
    });

    if (effortFilter === 'best') {
      const bestByWorkflow: Record<string, typeof allData[0]> = {};
      filteredByWorkflow.forEach(c => {
        const parsed = parseConditionName(c.condition);
        const name = parsed.name;
        if (!bestByWorkflow[name] || c.planned_accuracy > bestByWorkflow[name].planned_accuracy) {
          bestByWorkflow[name] = c;
        }
      });
      return Object.values(bestByWorkflow).sort((a, b) => b.planned_accuracy - a.planned_accuracy);
    } else {
      return [...filteredByWorkflow].sort((a, b) => b.planned_accuracy - a.planned_accuracy);
    }
  }, [getFilteredAccuracies, effortFilter, selectedWorkflows]);

  // Get dynamic paired comparisons based on baseline condition
  const getDynamicPairedComparisons = () => {
    if (!details) return [];
    
    const matchedTaskIds = new Set(
      details.tasks
        .filter(t => (filterSubject === 'all' || t.category === filterSubject) && (filterAnswerType === 'all' || t.answer_type === filterAnswerType))
        .map(t => t.id)
    );

    const baseRuns = details.runs.filter(r => r.condition === baselineCondition && matchedTaskIds.has(r.task_id) && r.correct !== null);
    const baseMap = new Map(baseRuns.map(r => [`${r.task_id}|${r.repetition}`, r.correct]));

    const conditions = Array.from(new Set(details.runs.map(r => r.condition))).filter(c => c !== baselineCondition);

    return conditions.map(cond => {
      const condRuns = details.runs.filter(r => r.condition === cond && matchedTaskIds.has(r.task_id) && r.correct !== null);
      
      let matchedCount = 0;
      let correctBase = 0;
      let correctCond = 0;
      const diffs_by_task: Record<string, number[]> = {};

      condRuns.forEach(r => {
        const key = `${r.task_id}|${r.repetition}`;
        const baseCorrect = baseMap.get(key);
        if (baseCorrect !== undefined) {
          matchedCount++;
          if (baseCorrect === true) correctBase++;
          if (r.correct === true) correctCond++;
          
          if (!diffs_by_task[r.task_id]) diffs_by_task[r.task_id] = [];
          const valBase = baseCorrect === true ? 1 : 0;
          const valCond = r.correct === true ? 1 : 0;
          diffs_by_task[r.task_id].push(valCond - valBase);
        }
      });

      // Simple bootstrap calculation in JS if needed, or fallback to backend computed ones if matches baseline.
      // To ensure statistical rigour, we compute bootstrap using the seeded algorithm in JS here:
      const ci = calculateBootstrapJS(diffs_by_task);

      return {
        condition: cond,
        baseline: baselineCondition,
        matched_pairs: matchedCount,
        accuracy_baseline: matchedCount > 0 ? correctBase / matchedCount : 0,
        accuracy_cond: matchedCount > 0 ? correctCond / matchedCount : 0,
        delta: matchedCount > 0 ? (correctCond - correctBase) / matchedCount : 0,
        ci_lower: ci.lower,
        ci_upper: ci.upper,
      };
    });
  };

  const calculateBootstrapJS = (diffs_by_task: Record<string, number[]>, replicatesCount = 1000) => {
    const taskIds = Object.keys(diffs_by_task);
    if (taskIds.length === 0) return { lower: 0, upper: 0 };
    
    // We want a stable seed-like random generator for reproducibility
    let seed = 42;
    const rng = () => {
      const x = Math.sin(seed++) * 10000;
      return x - Math.floor(x);
    };

    const replicates: number[] = [];
    for (let r = 0; r < replicatesCount; r++) {
      let totalDiff = 0;
      let totalCount = 0;
      for (let i = 0; i < taskIds.length; i++) {
        // Draw with replacement
        const idx = Math.floor(rng() * taskIds.length);
        const vals = diffs_by_task[taskIds[idx]];
        for (let j = 0; j < vals.length; j++) {
          totalDiff += vals[j];
          totalCount++;
        }
      }
      replicates.push(totalCount > 0 ? totalDiff / totalCount : 0);
    }
    
    replicates.sort((a, b) => a - b);
    return {
      lower: replicates[Math.floor(replicatesCount * 0.025)],
      upper: replicates[Math.floor(replicatesCount * 0.975)],
    };
  };

  const renderEfficiencyChart = () => {
    if (!details) return null;
    const allData = getFilteredAccuracies();
    // Do not filter out unselected conditions from rendering to keep axis stable and allow clicking them
    const data = allData;

    const maxCost = Math.max(...data.map(c => c.cost / (c.expected || 1)), 0.01);
    const maxTime = Math.max(...data.map(c => c.avg_time / 1000), 1.0);
    const maxTokens = Math.max(...data.map(c => c.output_tokens / (c.expected || 1)), 100);

    const getVal = (c: any) => {
      if (efficiencyMetric === 'cost') {
        return c.expected > 0 ? c.cost / c.expected : 0;
      } else if (efficiencyMetric === 'time') {
        return c.avg_time / 1000;
      } else {
        return c.expected > 0 ? c.output_tokens / c.expected : 0;
      }
    };

    const maxVal = efficiencyMetric === 'cost' ? maxCost : efficiencyMetric === 'time' ? maxTime : maxTokens;

    const formatTick = (val: number) => {
      if (efficiencyMetric === 'cost') {
        return `$${val.toFixed(2)}`;
      } else if (efficiencyMetric === 'time') {
        return `${val.toFixed(1)}s`;
      } else {
        return val.toLocaleString(undefined, { maximumFractionDigits: 0 });
      }
    };

    const xAxisLabel = efficiencyMetric === 'cost'
      ? 'Avg cost per task'
      : efficiencyMetric === 'time'
        ? 'Avg wall time per task'
        : 'Avg output tokens per task';

    const workflowColors: Record<string, { stroke: string; fill: string; text: string }> = {
      'solo': { stroke: '#2563eb', fill: '#3b82f6', text: '#1d4ed8' }, // blue
      'debate': { stroke: '#dc2626', fill: '#ef4444', text: '#b91c1c' }, // red
      'self-critic': { stroke: '#7c3aed', fill: '#8b5cf6', text: '#6d28d9' }, // purple
      'sample': { stroke: '#d97706', fill: '#f59e0b', text: '#b45309' }, // amber
      'supervisor': { stroke: '#16a34a', fill: '#4ade80', text: '#15803d' }, // green
      'unknown': { stroke: '#4b5563', fill: '#9ca3af', text: '#374151' } // gray
    };

    const selectedData = data.filter(c => selectedGroups.includes(getGroupName(c)));

    const seriesGroups: Record<string, typeof data> = {};
    selectedData.forEach(c => {
      const seriesKey = getSeriesKey(c);
      if (!seriesGroups[seriesKey]) seriesGroups[seriesKey] = [];
      seriesGroups[seriesKey].push(c);
    });

    const workflowGroups: Record<string, typeof data> = {};
    selectedData.forEach(c => {
      const workflow = getGroupName(c);
      if (!workflowGroups[workflow]) workflowGroups[workflow] = [];
      workflowGroups[workflow].push(c);
    });
    const groupRepresentatives = Object.entries(workflowGroups).map(([workflow, members]) => ({
      workflow,
      condition: [...members].sort((a, b) =>
        b.planned_accuracy - a.planned_accuracy || getVal(a) - getVal(b)
      )[0],
    }));

    const maxDataAcc = Math.max(...data.map(c => c.planned_accuracy), 0.1);
    let maxY = 1.0;
    if (maxDataAcc <= 0.4) {
      maxY = 0.6;
    } else if (maxDataAcc <= 0.5) {
      maxY = 0.7;
    } else if (maxDataAcc <= 0.7) {
      maxY = 0.8;
    }

    // Coordinate mapping helpers (Scaled 1.5x to match 1050x900 viewBox)
    const getX = (val: number) => {
      // Reversed X axis: 0 is at right (975), maxVal is at left (155) (leaving 50px space on the left)
      return 975 - (val / (maxVal || 1)) * 820;
    };
    const getY = (acc: number) => {
      // Y axis: 0% is at bottom (555), maxY% is at top (55) (Scaled down by 30% vertically)
      return 555 - (acc / maxY) * 500;
    };

    // Find hovered point coordinates if any
    const hoveredPoint = data.find(c => c.condition === hoveredCondition);
    const hoveredVal = hoveredPoint ? getVal(hoveredPoint) : null;
    const hoveredX = hoveredPoint && hoveredVal !== null ? getX(hoveredVal) : null;
    const hoveredY = hoveredPoint ? getY(hoveredPoint.planned_accuracy) : null;
    const hoveredColors = hoveredPoint ? (workflowColors[getGroupName(hoveredPoint)] || workflowColors['unknown']) : null;

    const allGroups = Array.from(new Set(data.map(c => getGroupName(c)))).sort((a, b) => {
      const aIndex = WORKFLOW_ORDER.indexOf(a);
      const bIndex = WORKFLOW_ORDER.indexOf(b);
      return (aIndex === -1 ? 99 : aIndex) - (bIndex === -1 ? 99 : bIndex);
    });

    const tooltipWidth = 238;
    const tooltipX = hoveredX === null ? 0 : hoveredX > 720 ? hoveredX - tooltipWidth - 18 : hoveredX + 18;
    const tooltipY = hoveredY === null ? 0 : Math.max(62, Math.min(hoveredY - 38, 465));

    return (
      <div>
        {/* Selector Controls */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div style={{ display: 'flex', background: 'var(--bg-secondary)', borderRadius: '6px', padding: '3px', border: '1px solid var(--border-color)' }}>
            <button
              onClick={() => setEfficiencyMetric('cost')}
              style={{
                background: efficiencyMetric === 'cost' ? 'var(--bg-primary)' : 'transparent',
                color: efficiencyMetric === 'cost' ? 'var(--text-primary)' : 'var(--text-secondary)',
                border: efficiencyMetric === 'cost' ? '1px solid var(--border-color)' : 'none',
                borderRadius: '4px',
                padding: '6px 12px',
                fontSize: '11px',
                fontWeight: '600',
                cursor: 'pointer',
                boxShadow: efficiencyMetric === 'cost' ? '0 1px 2px rgba(0,0,0,0.05)' : 'none',
              }}
            >
              Cost
            </button>
            <button
              onClick={() => setEfficiencyMetric('time')}
              style={{
                background: efficiencyMetric === 'time' ? 'var(--bg-primary)' : 'transparent',
                color: efficiencyMetric === 'time' ? 'var(--text-primary)' : 'var(--text-secondary)',
                border: efficiencyMetric === 'time' ? '1px solid var(--border-color)' : 'none',
                borderRadius: '4px',
                padding: '6px 12px',
                fontSize: '11px',
                fontWeight: '600',
                cursor: 'pointer',
                boxShadow: efficiencyMetric === 'time' ? '0 1px 2px rgba(0,0,0,0.05)' : 'none',
              }}
            >
              Time
            </button>
            <button
              onClick={() => setEfficiencyMetric('tokens')}
              style={{
                background: efficiencyMetric === 'tokens' ? 'var(--bg-primary)' : 'transparent',
                color: efficiencyMetric === 'tokens' ? 'var(--text-primary)' : 'var(--text-secondary)',
                border: efficiencyMetric === 'tokens' ? '1px solid var(--border-color)' : 'none',
                borderRadius: '4px',
                padding: '6px 12px',
                fontSize: '11px',
                fontWeight: '600',
                cursor: 'pointer',
                boxShadow: efficiencyMetric === 'tokens' ? '0 1px 2px rgba(0,0,0,0.05)' : 'none',
              }}
            >
              Output tokens
            </button>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '11px', color: 'var(--text-secondary)' }}>
            <span>Updated {details?.manifest?.created_at ? new Date(details.manifest.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }) : 'Recently'}</span>
            
            {/* Interactive Conditions Dropdown */}
            <div style={{ position: 'relative' }}>
              <button
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                style={{
                  background: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-primary)',
                  borderRadius: '4px',
                  padding: '6px 12px',
                  fontSize: '11px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  boxShadow: '0 1px 2px rgba(0,0,0,0.05)'
                }}
              >
                Workflows ({selectedGroups.length}/{allGroups.length}) {isDropdownOpen ? '▲' : '▼'}
              </button>
              {isDropdownOpen && (
                <>
                  <div
                    onClick={() => setIsDropdownOpen(false)}
                    style={{
                      position: 'fixed',
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      zIndex: 999,
                    }}
                  />
                  <div
                    style={{
                      position: 'absolute',
                      right: 0,
                      top: '100%',
                      marginTop: '4px',
                      background: 'var(--bg-primary)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '6px',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
                      padding: '12px',
                      zIndex: 1000,
                      minWidth: '220px',
                      maxHeight: '300px',
                      overflowY: 'auto',
                    }}
                  >
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {allGroups.map((wf: any) => {
                        const isChecked = selectedGroups.includes(wf);
                        return (
                          <label
                            key={wf}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '8px',
                              fontSize: '11px',
                              cursor: 'pointer',
                              color: 'var(--text-primary)',
                              padding: '4px 6px',
                              borderRadius: '4px',
                              transition: 'background 0.1s ease',
                              textTransform: 'capitalize',
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-secondary)'}
                            onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                          >
                            <input
                              type="checkbox"
                              checked={isChecked}
                              onChange={() => {
                                if (isChecked) {
                                  setSelectedGroups(selectedGroups.filter(sg => sg !== wf));
                                } else {
                                  setSelectedGroups([...selectedGroups, wf]);
                                }
                              }}
                              style={{ cursor: 'pointer' }}
                            />
                            <span>{formatWorkflowName(wf)}</span>
                          </label>
                        );
                      })}
                    </div>
                    <div style={{ borderTop: '1px solid var(--border-color)', marginTop: '8px', paddingTop: '8px', display: 'flex', gap: '8px' }}>
                      <button
                        onClick={() => setSelectedGroups(allGroups)}
                        style={{
                          background: 'var(--accent-light)',
                          border: 'none',
                          borderRadius: '4px',
                          padding: '4px 8px',
                          fontSize: '10px',
                          fontWeight: '600',
                          cursor: 'pointer',
                          color: 'var(--text-primary)',
                        }}
                      >
                        Select all
                      </button>
                      <button
                        onClick={() => setSelectedGroups([])}
                        style={{
                          background: 'var(--accent-light)',
                          border: 'none',
                          borderRadius: '4px',
                          padding: '4px 8px',
                          fontSize: '10px',
                          fontWeight: '600',
                          cursor: 'pointer',
                          color: 'var(--text-primary)',
                        }}
                      >
                        Clear
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Chart Area */}
        <div style={{ width: '100%', overflow: 'hidden', position: 'relative' }}>
          <svg viewBox="0 0 1050 630" width="100%" height="630" style={{ background: 'var(--bg-primary)' }}>
            {/* Grid labels for Y (Accuracy) */}
            {Array.from({ length: Math.round(maxY * 10) + 1 }, (_, i) => i / 10).map((val, idx) => {
              const y = getY(val);
              // Skip rendering tick label if it's close to the hovered Y accuracy to prevent overlapping
              const isOverlapped = hoveredY !== null && Math.abs(getY(val) - hoveredY) < 18;
              return (
                <g key={idx}>
                  <line x1="105" y1={y} x2="975" y2={y} stroke="#f3f4f6" strokeWidth="1" />
                  {!isOverlapped && (
                    <text x="90" y={y + 4} textAnchor="end" fill="var(--text-secondary)" style={{ fontSize: '12px' }}>
                      {(val * 100).toFixed(0)}%
                    </text>
                  )}
                </g>
              );
            })}
            
            {/* Grid labels for X (Metric) */}
            {[0, 0.25, 0.5, 0.75, 1.0].map((val, idx) => {
              const tickVal = val * maxVal;
              const x = getX(tickVal);
              const isOverlapped = hoveredX !== null && Math.abs(x - hoveredX) < 45;
              return (
                <g key={idx}>
                  <line x1={x} y1="55" x2={x} y2="555" stroke="#f3f4f6" strokeWidth="1" />
                  {!isOverlapped && (
                    <text x={x} y="575" textAnchor="middle" fill="var(--text-secondary)" style={{ fontSize: '12px' }}>
                      {formatTick(tickVal)}
                    </text>
                  )}
                </g>
              );
            })}

            {/* Axis Lines */}
            <line x1="105" y1="55" x2="105" y2="555" stroke="var(--border-color)" strokeWidth="1" />
            <line x1="105" y1="555" x2="975" y2="555" stroke="var(--border-color)" strokeWidth="1" />

            {/* Chart Title */}
            <text x="105" y="35" fill="var(--text-primary)" style={{ fontSize: '18px', fontWeight: '700' }}>
              Planned-job accuracy
            </text>

            {/* most efficient label */}
            <text x="975" y="35" textAnchor="end" fill="var(--text-secondary)" style={{ fontSize: '14px', fontStyle: 'italic' }}>
              most efficient ↗
            </text>

            {/* Axis Title */}
            <text x="540" y="615" textAnchor="middle" fill="var(--text-primary)" style={{ fontSize: '16px', fontWeight: '600' }}>
              {xAxisLabel}
            </text>

            {/* Connect only configurations where one scale variable changes. */}
            {Object.entries(seriesGroups).map(([seriesKey, members]) => {
              const sorted = [...members].sort((a, b) => getVal(a) - getVal(b));
              if (sorted.length < 2) return null;
              const pathData = sorted.map((c, idx) => {
                const x = getX(getVal(c));
                const y = getY(c.planned_accuracy);
                return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
              }).join(' ');
              const wf = getGroupName(members[0]);
              const colors = workflowColors[wf] || workflowColors['unknown'];
              return (
                <path
                  key={seriesKey}
                  d={pathData}
                  fill="none"
                  stroke={colors.stroke}
                  strokeWidth={wf === 'supervisor' ? "2.4" : "2"}
                  style={{ opacity: 0.72 }}
                />
              );
            })}

            {/* Render dashed lines and highlights for hovered condition */}
            {hoveredPoint && hoveredX !== null && hoveredY !== null && hoveredColors && hoveredVal !== null && (
              <g>
                {/* Dashed line to Y axis */}
                <line x1="105" y1={hoveredY} x2={hoveredX} y2={hoveredY} stroke={hoveredColors.stroke} strokeWidth="1.2" strokeDasharray="3,3" />
                {/* Dashed line to X axis */}
                <line x1={hoveredX} y1={hoveredY} x2={hoveredX} y2="555" stroke={hoveredColors.stroke} strokeWidth="1.2" strokeDasharray="3,3" />
                
                {/* Highlighted tick label on Y-axis */}
                <text x="90" y={hoveredY + 4} textAnchor="end" fill={hoveredColors.text} style={{ fontSize: '12px', fontWeight: 'bold' }}>
                  {(hoveredPoint.planned_accuracy * 100).toFixed(0)}%
                </text>
                <line x1="99" y1={hoveredY} x2="105" y2={hoveredY} stroke={hoveredColors.stroke} strokeWidth="1.5" />

                {/* Highlighted tick label on X-axis */}
                <text x={hoveredX} y="575" textAnchor="middle" fill={hoveredColors.text} style={{ fontSize: '12px', fontWeight: 'bold' }}>
                  {formatTick(hoveredVal)}
                </text>
                <line x1={hoveredX} y1="555" x2={hoveredX} y2="561" stroke={hoveredColors.stroke} strokeWidth="1.5" />
                
                {/* Highlight ring around hovered circle */}
                <circle cx={hoveredX} cy={hoveredY} r="12.5" fill="none" stroke={hoveredColors.stroke} strokeWidth="1.2" />
              </g>
            )}

            {/* Plot points and condition labels */}
            {selectedData.map((c, idx) => {
              const val = getVal(c);
              const x = getX(val);
              const y = getY(c.planned_accuracy);
              const colors = workflowColors[getGroupName(c)] || workflowColors['unknown'];

              const isHovered = hoveredCondition === c.condition;

              return (
                <g 
                  key={idx}
                  style={{ 
                    opacity: isHovered ? 1.0 : 0.9, 
                    transition: 'opacity 0.15s ease',
                    cursor: 'pointer' 
                  }}
                >
                  <circle 
                    cx={x} 
                    cy={y} 
                    r={isHovered ? "10" : "7.5"} 
                    fill={colors.fill} 
                    stroke={isHovered ? colors.stroke : "white"} 
                    strokeWidth={isHovered ? "3" : "2"} 
                  >
                    <title>{formatWorkflowName(getGroupName(c))}: {formatConditionConfig(c)}</title>
                  </circle>
                  
                  {/* Larger invisible overlay for easier mouse hovering and clicking */}
                  <circle
                    cx={x}
                    cy={y}
                    r="15"
                    fill="transparent"
                    style={{ cursor: 'pointer', pointerEvents: 'all' }}
                    onMouseEnter={() => setHoveredCondition(c.condition)}
                    onMouseLeave={() => setHoveredCondition(null)}
                    onClick={() => {
                      const gName = getGroupName(c);
                      if (selectedGroups.includes(gName)) {
                        setSelectedGroups(selectedGroups.filter(sg => sg !== gName));
                      } else {
                        setSelectedGroups([...selectedGroups, gName]);
                      }
                    }}
                  />
                </g>
              );
            })}

            {/* Annotate the highest-accuracy condition in each workflow group. */}
            {groupRepresentatives.map(({ workflow, condition }) => {
              if (!condition) return null;
              const pointX = getX(getVal(condition));
              const pointY = getY(condition.planned_accuracy);
              const colors = workflowColors[workflow] || workflowColors['unknown'];
              const label = formatConditionTag(condition);
              const tagWidth = Math.min(238, Math.max(94, label.length * 6.15 + 22));
              const tagX = pointX > 800
                ? pointX - tagWidth - 12
                : pointX < 260
                  ? pointX + 12
                  : pointX - tagWidth / 2;
              const tagY = Math.max(62, pointY - 38);
              const leaderX = Math.max(tagX + 12, Math.min(pointX, tagX + tagWidth - 12));

              return (
                <g key={`tag-${workflow}`} pointerEvents="none">
                  <line
                    x1={pointX}
                    y1={pointY - 9}
                    x2={leaderX}
                    y2={tagY + 24}
                    stroke={colors.stroke}
                    strokeWidth="1"
                    opacity="0.45"
                  />
                  <rect
                    x={tagX}
                    y={tagY}
                    width={tagWidth}
                    height="24"
                    rx="4"
                    fill="white"
                    stroke={colors.stroke}
                    strokeOpacity="0.3"
                  />
                  <circle cx={tagX + 11} cy={tagY + 12} r="3.5" fill={colors.fill} />
                  <text
                    x={tagX + 20}
                    y={tagY + 15.5}
                    fill={colors.text}
                    style={{ fontSize: '10px', fontWeight: '600', fontFamily: 'var(--font-mono)' }}
                  >
                    {label}
                  </text>
                </g>
              );
            })}

            {/* Hover details use the same compact treatment as the group tags. */}
            {hoveredPoint && hoveredX !== null && hoveredY !== null && hoveredColors && hoveredVal !== null && (
              <g pointerEvents="none">
                <rect
                  x={tooltipX}
                  y={tooltipY}
                  width={tooltipWidth}
                  height="68"
                  rx="4"
                  fill="white"
                  stroke={hoveredColors.stroke}
                  strokeOpacity="0.35"
                  style={{ filter: 'drop-shadow(0 2px 5px rgba(15, 23, 42, 0.09))' }}
                />
                <circle cx={tooltipX + 12} cy={tooltipY + 15} r="3.5" fill={hoveredColors.fill} />
                <text x={tooltipX + 21} y={tooltipY + 18.5} fill={hoveredColors.text} style={{ fontSize: '10px', fontWeight: '700', fontFamily: 'var(--font-mono)' }}>
                  {formatConditionTag(hoveredPoint)}
                </text>
                <text x={tooltipX + 12} y={tooltipY + 38} fill="var(--text-secondary)" style={{ fontSize: '9.5px' }}>
                  {formatConditionConfig(hoveredPoint)}
                </text>
                <text x={tooltipX + 12} y={tooltipY + 57} fill="var(--text-primary)" style={{ fontSize: '10.5px', fontWeight: '600' }}>
                  {(hoveredPoint.planned_accuracy * 100).toFixed(1)}% accuracy
                </text>
                <text x={tooltipX + tooltipWidth - 12} y={tooltipY + 57} textAnchor="end" fill="var(--text-secondary)" style={{ fontSize: '10.5px' }}>
                  {formatTick(hoveredVal)}
                </text>
              </g>
            )}
          </svg>

          {/* Legend */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '15px', justifyContent: 'center', marginTop: '16px' }}>
            {allGroups.filter(group => selectedGroups.includes(group)).map(group => {
              const wf = group;
              const colors = workflowColors[wf] || workflowColors['unknown'];
              return (
                <div key={group} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px' }}>
                  <span style={{ display: 'inline-block', width: '14px', height: '2px', backgroundColor: colors.stroke }} />
                  <span style={{ display: 'inline-block', width: '7px', height: '7px', borderRadius: '50%', backgroundColor: colors.fill, border: `1px solid ${colors.stroke}`, marginLeft: '-10px' }} />
                  <span style={{ fontWeight: '600', color: 'var(--text-primary)' }}>{formatWorkflowName(group)}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>MART Local Results Dashboard</h1>
        {view !== 'list' && (
          <button className="btn btn-sm" onClick={() => { setView('list'); setDetails(null); }}>
            ← Back to Overview
          </button>
        )}
      </header>

      <main className="main-content">
        {loading && (
          <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-secondary)' }}>
            Loading experiment data...
          </div>
        )}

        {errorMsg && (
          <div style={{ padding: '16px', background: '#fef2f2', border: '1px solid #fee2e2', color: '#b91c1c', borderRadius: '4px', marginBottom: '24px' }}>
            <strong>Error</strong>: {errorMsg}
          </div>
        )}

        {/* 1. LIST VIEW */}
        {view === 'list' && !loading && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h2 style={{ fontSize: '15px', fontWeight: '600' }}>Discovered Experiments</h2>
              {selectedRunsToCompare.length >= 2 && (
                <button className="btn btn-primary" onClick={handleCompare}>
                  Compare Selected ({selectedRunsToCompare.length})
                </button>
              )}
            </div>

            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th style={{ width: '40px' }}></th>
                    <th>Experiment ID & Purpose</th>
                    <th>Status</th>
                    <th>Model</th>
                    <th>Grading Model</th>
                    <th>Benchmark</th>
                    <th style={{ textAlign: 'right' }}>Tasks</th>
                    <th style={{ textAlign: 'right' }}>Completed/Planned</th>
                    <th style={{ textAlign: 'right' }}>Graded/Planned</th>
                    <th style={{ textAlign: 'right' }}>Best Condition (Binomial Accuracy)</th>
                    <th style={{ textAlign: 'right' }}>Total Cost</th>
                    <th>Updated At</th>
                  </tr>
                </thead>
                <tbody>
                  {experiments.map(exp => (
                    <tr key={exp.experiment_id}>
                      <td style={{ textAlign: 'center' }}>
                        <input
                          type="checkbox"
                          checked={selectedRunsToCompare.includes(exp.experiment_id)}
                          onChange={() => handleToggleSelectCompare(exp.experiment_id)}
                        />
                      </td>
                      <td>
                        <div style={{ fontWeight: '600', color: 'var(--accent-highlight)', cursor: 'pointer' }} onClick={() => handleOpenDetail(exp.experiment_id)}>
                          {exp.experiment_id}
                        </div>
                        <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                          {exp.purpose}
                        </div>
                      </td>
                      <td>
                        <span className={`badge badge-${exp.status}`}>
                          {exp.status.replace('_', ' ')}
                        </span>
                      </td>
                      <td>{exp.model}</td>
                      <td>{exp.grader_model}</td>
                      <td>{exp.task_set}</td>
                      <td style={{ textAlign: 'right' }}>{exp.task_count}</td>
                      <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                        {exp.completed_jobs} / {exp.expected_jobs}
                      </td>
                      <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                        {exp.graded_jobs} / {exp.expected_jobs}
                      </td>
                      <td style={{ textAlign: 'right' }}>
                        {exp.best_condition ? (
                          <div>
                            <span style={{ fontWeight: '600' }}>
                              {(exp.best_accuracy! * 100).toFixed(1)}%
                            </span>
                            <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: '4px' }}>
                              [{((exp.best_ci_lower || 0) * 100).toFixed(0)} - {((exp.best_ci_upper || 0) * 100).toFixed(0)}%]
                            </span>
                            <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>
                              {exp.best_condition}
                            </div>
                          </div>
                        ) : 'n/a'}
                      </td>
                      <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                        {formatUSD(exp.total_cost)}
                      </td>
                      <td style={{ fontSize: '12px' }}>{formatTime(exp.updated_at)}</td>
                    </tr>
                  ))}
                  {experiments.length === 0 && (
                    <tr>
                      <td colSpan={12} style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
                        No experiments found in results/ directory.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* 2. COMPARE VIEW */}
        {view === 'compare' && comparisonResult && !loading && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h2 style={{ fontSize: '15px', fontWeight: '600' }}>Cross-Experiment Comparison</h2>
              <button className="btn btn-sm" onClick={() => setView('list')}>✕ Close</button>
            </div>

            {/* Warnings/Comparability checks */}
            {!comparisonResult.comparable && (
              <div style={{ padding: '16px', background: '#fffbeb', border: '1px solid #fef3c7', borderRadius: '4px', marginBottom: '24px' }}>
                <strong style={{ color: '#b45309' }}>⚠ Scientific Comparability Warning</strong>
                <p style={{ fontSize: '12px', color: '#78350f', marginTop: '4px' }}>
                  The selected experiments differ in key variables. Delta calculations have been disabled to prevent misleading scientific conclusions:
                </p>
                <ul style={{ fontSize: '11px', color: '#78350f', paddingLeft: '20px', marginTop: '8px' }}>
                  {comparisonResult.warnings.map((w: string, idx: number) => (
                    <li key={idx} style={{ marginBottom: '4px' }}>{w}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Side by side stats */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', marginBottom: '24px' }}>
              {comparisonResult.experiments.map((exp: any) => (
                <div key={exp.id} style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '16px' }}>
                  <h3 style={{ fontSize: '14px', fontWeight: '700', marginBottom: '8px' }}>{exp.id}</h3>
                  <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '12px' }}>{exp.purpose}</div>
                  <table style={{ fontSize: '11px' }}>
                    <thead>
                      <tr>
                        <th>Condition</th>
                        <th style={{ textAlign: 'right' }}>Accuracy</th>
                        <th style={{ textAlign: 'right' }}>Cost</th>
                      </tr>
                    </thead>
                    <tbody>
                      {exp.conditions.map((c: any) => (
                        <tr key={c.condition}>
                          <td>{c.condition}</td>
                          <td style={{ textAlign: 'right', fontWeight: '600' }}>{(c.planned_job_accuracy * 100).toFixed(1)}%</td>
                          <td style={{ textAlign: 'right' }}>{formatUSD(c.cost_usd)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>

            {/* Comparable Deltas table */}
            {comparisonResult.comparable && (
              <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '16px' }}>
                <h3 style={{ fontSize: '14px', fontWeight: '700', marginBottom: '12px' }}>Paired Deltas across Matched Tasks</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Experiment A</th>
                      <th>Experiment B</th>
                      <th>Condition</th>
                      <th style={{ textAlign: 'right' }}>Matched Pairs</th>
                      <th style={{ textAlign: 'right' }}>Acc A</th>
                      <th style={{ textAlign: 'right' }}>Acc B</th>
                      <th style={{ textAlign: 'right' }}>Delta (B - A)</th>
                      <th style={{ textAlign: 'right' }}>95% Task-Clustered CI</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparisonResult.paired_deltas.map((d: any, idx: number) => (
                      <tr key={idx}>
                        <td>{d.experiment_a}</td>
                        <td>{d.experiment_b}</td>
                        <td>{d.condition}</td>
                        <td style={{ textAlign: 'right' }}>{d.matched_pairs}</td>
                        <td style={{ textAlign: 'right' }}>{(d.accuracy_a * 100).toFixed(1)}%</td>
                        <td style={{ textAlign: 'right' }}>{(d.accuracy_b * 100).toFixed(1)}%</td>
                        <td style={{ textAlign: 'right', fontWeight: '600', color: d.accuracy_delta > 0 ? 'var(--success-color)' : d.accuracy_delta < 0 ? 'var(--error-color)' : 'inherit' }}>
                          {d.accuracy_delta > 0 ? '+' : ''}{(d.accuracy_delta * 100).toFixed(1)}%
                        </td>
                        <td style={{ textAlign: 'right', color: 'var(--text-secondary)' }}>
                          [{((d.ci_lower || 0) * 100).toFixed(1)}%, {((d.ci_upper || 0) * 100).toFixed(1)}%]
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* 3. DETAIL VIEW */}
        {view === 'detail' && details && !loading && (
          <div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <h2 style={{ fontSize: '16px', fontWeight: '700' }}>{selectedExpId}</h2>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '12px', marginTop: '4px' }}>
                    {details.manifest.metadata?.purpose || 'No purpose description available'}
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '12px' }}>
                  {/* Subject filter */}
                  <div>
                    <label style={{ fontSize: '11px', fontWeight: '600', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Subject Filter</label>
                    <select className="form-input" style={{ width: '150px' }} value={filterSubject} onChange={(e) => setFilterSubject(e.target.value)}>
                      <option value="all">All Subjects</option>
                      {getSubjectsList().map(sub => (
                        <option key={sub} value={sub}>{sub}</option>
                      ))}
                    </select>
                  </div>
                  {/* Answer type filter */}
                  <div>
                    <label style={{ fontSize: '11px', fontWeight: '600', color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Answer Type</label>
                    <select className="form-input" style={{ width: '130px' }} value={filterAnswerType} onChange={(e) => setFilterAnswerType(e.target.value)}>
                      <option value="all">All Types</option>
                      <option value="multiple_choice">Multiple Choice</option>
                      <option value="short_answer">Short Answer</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>

            {/* Warnings list */}
            {details.warnings.length > 0 && (
              <div className="warnings-banner">
                {details.warnings.map((w, idx) => (
                  <div key={idx} className="warning-item">
                    <span style={{ marginRight: '6px' }}>⚠</span>
                    <span>{w}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Tabs Navigation */}
            <div className="tab-list">
              <button className={`tab-btn ${activeTab === 'results' ? 'active' : ''}`} onClick={() => setActiveTab('results')}>Primary Results</button>
              <button className={`tab-btn ${activeTab === 'protocol' ? 'active' : ''}`} onClick={() => setActiveTab('protocol')}>Protocol</button>
              <button className={`tab-btn ${activeTab === 'health' ? 'active' : ''}`} onClick={() => setActiveTab('health')}>Execution Health</button>
              <button className={`tab-btn ${activeTab === 'efficiency' ? 'active' : ''}`} onClick={() => setActiveTab('efficiency')}>Efficiency</button>
              <button className={`tab-btn ${activeTab === 'per_question' ? 'active' : ''}`} onClick={() => setActiveTab('per_question')}>Per-Question Analysis</button>
              <button className={`tab-btn ${activeTab === 'multi_agent' ? 'active' : ''}`} onClick={() => setActiveTab('multi_agent')}>Multi-Agent Behavior</button>
              <button className={`tab-btn ${activeTab === 'runs' ? 'active' : ''}`} onClick={() => setActiveTab('runs')}>Job Details</button>
            </div>

            {/* TAB CONTENTS */}

            {/* 3A. PROTOCOL */}
            {activeTab === 'protocol' && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '20px' }}>
                  <h3 style={{ fontSize: '13px', fontWeight: '750', textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: '12px', borderBottom: '1px solid var(--border-color)', paddingBottom: '4px' }}>Context</h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    <div>
                      <strong style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Experiment ID:</strong>
                      <div style={{ fontSize: '13px', fontFamily: 'var(--font-mono)' }}>{details.manifest.experiment_id}</div>
                    </div>
                    <div>
                      <strong style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Research Purpose:</strong>
                      <div style={{ fontSize: '13px' }}>{details.manifest.metadata?.purpose || 'n/a'}</div>
                    </div>
                    <div>
                      <strong style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Primary Generation Model:</strong>
                      <div style={{ fontSize: '13px' }}>{details.manifest.model}</div>
                    </div>
                    <div>
                      <strong style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Judge Model / Tie Breaker:</strong>
                      <div style={{ fontSize: '13px' }}>{details.manifest.judge_model || 'n/a'}</div>
                    </div>
                    <div>
                      <strong style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Configuration Fingerprint:</strong>
                      <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)' }}>{details.manifest.compatibility_fingerprint || 'n/a'}</div>
                    </div>
                    <div>
                      <strong style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Timestamps:</strong>
                      <div style={{ fontSize: '12px' }}>
                        Created: {formatTime(details.manifest.created_at)}<br />
                        Last Updated: {formatTime(details.ledger?.updated_at || details.manifest.created_at)}
                      </div>
                    </div>
                  </div>
                </div>

                <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '20px' }}>
                  <h3 style={{ fontSize: '13px', fontWeight: '750', textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: '12px', borderBottom: '1px solid var(--border-color)', paddingBottom: '4px' }}>Task Set & Concurrency</h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    <div>
                      <strong style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Task Set Path:</strong>
                      <div style={{ fontSize: '13px' }}>{details.manifest.task_set_path}</div>
                    </div>
                    <div>
                      <strong style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Task Set SHA-256:</strong>
                      <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)' }}>{details.manifest.task_set_sha256}</div>
                    </div>
                    <div>
                      <strong style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Task Composition:</strong>
                      <div style={{ fontSize: '12px' }}>
                        Total Questions: {details.manifest.task_count}<br />
                        Repetitions per condition: {details.manifest.repetitions}<br />
                        Conditions evaluated: {details.manifest.conditions.length}
                      </div>
                    </div>
                    <div>
                      <strong style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Rate Limits & Concurrency:</strong>
                      <div style={{ fontSize: '12px' }}>
                        Concurrency: {details.manifest.policy?.concurrency || 1}<br />
                        Max In-Flight: {details.manifest.policy?.max_in_flight_requests || 1}<br />
                        Requests Per Minute: {details.manifest.policy?.requests_per_minute || 'Unlimited'}<br />
                        Tokens Per Minute: {details.manifest.policy?.tokens_per_minute || 'Unlimited'}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 3B. EXECUTION HEALTH */}
            {activeTab === 'health' && (
              <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '20px' }}>
                <h3 style={{ fontSize: '14px', fontWeight: '700', marginBottom: '16px' }}>Execution Coverage and Denominators</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Condition</th>
                      <th style={{ textAlign: 'right' }}>Planned Jobs</th>
                      <th style={{ textAlign: 'right' }}>Valid Completed</th>
                      <th style={{ textAlign: 'right' }}>Coverage Rate</th>
                      <th style={{ textAlign: 'right' }}>Contract Invalid</th>
                      <th style={{ textAlign: 'right' }}>Execution Failures</th>
                      <th style={{ textAlign: 'right' }}>Inconclusive</th>
                      <th style={{ textAlign: 'right' }}>Missing</th>
                      <th style={{ textAlign: 'right' }}>Attempts / Retries</th>
                    </tr>
                  </thead>
                  <tbody>
                    {details.summary.conditions.map(c => (
                      <tr key={c.condition}>
                        <td style={{ fontWeight: '600' }}>{c.condition}</td>
                        <td style={{ textAlign: 'right' }}>{c.expected_jobs}</td>
                        <td style={{ textAlign: 'right' }}>{c.completed_answer_jobs}</td>
                        <td style={{ textAlign: 'right', fontWeight: '600' }}>{(c.coverage_rate * 100).toFixed(1)}%</td>
                        <td style={{ textAlign: 'right', color: c.contract_invalid_outputs > 0 ? 'var(--warning-color)' : 'inherit' }}>
                          {c.contract_invalid_outputs}
                        </td>
                        <td style={{ textAlign: 'right', color: c.provider_execution_failures > 0 ? 'var(--error-color)' : 'inherit' }}>
                          {c.provider_execution_failures}
                        </td>
                        <td style={{ textAlign: 'right', color: c.inconclusive_jobs > 0 ? 'var(--inconclusive-color)' : 'inherit' }}>
                          {c.inconclusive_jobs}
                        </td>
                        <td style={{ textAlign: 'right', color: c.missing_jobs > 0 ? 'var(--text-muted)' : 'inherit' }}>
                          {c.missing_jobs}
                        </td>
                        <td style={{ textAlign: 'right' }}>
                          {c.attempts} <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>({c.retried_jobs} retries)</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* 3C. PRIMARY RESULTS */}
            {activeTab === 'results' && (
              <div>
                {/* Accuracy comparison table */}
                <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '20px', marginBottom: '24px' }}>
                  <h3 style={{ fontSize: '14px', fontWeight: '700', marginBottom: '12px' }}>Condition Binomial Accuracy</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>Condition</th>
                        <th style={{ textAlign: 'right' }}>Planned-Job Accuracy</th>
                        <th style={{ textAlign: 'right' }}>95% Wilson CI</th>
                        <th style={{ textAlign: 'right' }}>Graded Accuracy</th>
                        <th style={{ textAlign: 'right' }}>95% Wilson Graded CI</th>
                        <th style={{ textAlign: 'right' }}>Numerator/Denominator</th>
                      </tr>
                    </thead>
                    <tbody>
                      {getFilteredAccuracies().map(c => (
                        <tr key={c.condition}>
                          <td style={{ fontWeight: '600' }}>{c.condition}</td>
                          <td style={{ textAlign: 'right', fontWeight: '700', fontSize: '14px' }}>
                            {(c.planned_accuracy * 100).toFixed(1)}%
                          </td>
                          <td style={{ textAlign: 'right', color: 'var(--text-secondary)' }}>
                            [{((c.planned_ci_lower || 0) * 100).toFixed(1)}% - {((c.planned_ci_upper || 0) * 100).toFixed(1)}%]
                          </td>
                          <td style={{ textAlign: 'right' }}>
                            {(c.graded_accuracy * 100).toFixed(1)}%
                          </td>
                          <td style={{ textAlign: 'right', color: 'var(--text-secondary)' }}>
                            [{((c.graded_ci_lower || 0) * 100).toFixed(1)}% - {((c.graded_ci_upper || 0) * 100).toFixed(1)}%]
                          </td>
                          <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                            {c.correct} / {c.expected}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* SVG Visualizations of Accuracy & CIs */}
                <div className="results-grid">
                  {/* Accuracy Bar Chart with Error Whiskers (Deepswe-style) */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '12px', marginBottom: '4px' }}>
                      {/* Effort level toggle buttons */}
                      <div style={{ display: 'inline-flex', height: '28px', alignItems: 'center', border: '1px solid var(--border-color)', borderRadius: '6px', overflow: 'hidden', fontSize: '12px' }}>
                        <button
                          type="button"
                          onClick={() => setEffortFilter('best')}
                          style={{
                            height: '100%',
                            padding: '0 12px',
                            border: 'none',
                            cursor: 'pointer',
                            fontWeight: 500,
                            backgroundColor: effortFilter === 'best' ? 'var(--accent-color)' : 'transparent',
                            color: effortFilter === 'best' ? '#ffffff' : 'var(--text-secondary)',
                            transition: 'all 0.15s ease',
                          }}
                        >
                          Best
                        </button>
                        <button
                          type="button"
                          onClick={() => setEffortFilter('all')}
                          style={{
                            height: '100%',
                            padding: '0 12px',
                            border: 'none',
                            borderLeft: '1px solid var(--border-color)',
                            cursor: 'pointer',
                            fontWeight: 500,
                            backgroundColor: effortFilter === 'all' ? 'var(--accent-color)' : 'transparent',
                            color: effortFilter === 'all' ? '#ffffff' : 'var(--text-secondary)',
                            transition: 'all 0.15s ease',
                          }}
                        >
                          All effort levels
                        </button>
                      </div>

                      {/* Workflows checklist dropdown */}
                      <div style={{ position: 'relative', display: 'inline-block' }}>
                        <button 
                          type="button" 
                          onClick={() => setShowWorkflowDropdown(!showWorkflowDropdown)}
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            border: '1px solid var(--border-color)',
                            backgroundColor: 'var(--bg-primary)',
                            cursor: 'pointer',
                            height: '28px',
                            gap: '6px',
                            padding: '0 10px',
                            borderRadius: '6px',
                            fontSize: '12px',
                            fontWeight: 500,
                            transition: 'all 0.15s ease',
                          }}
                          onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'; }}
                          onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'var(--bg-primary)'; }}
                        >
                          <span>Workflows</span>
                          <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontSize: '11px' }}>
                            ({selectedWorkflows.length}/{allWorkflows.length})
                          </span>
                          <svg 
                            xmlns="http://www.w3.org/2000/svg" 
                            width="14" 
                            height="14" 
                            viewBox="0 0 24 24" 
                            fill="none" 
                            stroke="currentColor" 
                            strokeWidth="2" 
                            strokeLinecap="round" 
                            strokeLinejoin="round"
                            style={{
                              transform: showWorkflowDropdown ? 'rotate(180deg)' : 'rotate(0deg)',
                              transition: 'transform 0.2s ease',
                            }}
                          >
                            <path d="m6 9 6 6 6-6" />
                          </svg>
                        </button>
                        
                        {showWorkflowDropdown && (
                          <>
                            <div 
                              style={{ position: 'fixed', inset: 0, zIndex: 998 }} 
                              onClick={() => setShowWorkflowDropdown(false)} 
                            />
                            <div 
                              style={{
                                position: 'absolute',
                                top: '32px',
                                right: 0,
                                backgroundColor: 'var(--bg-primary)',
                                border: '1px solid var(--border-color)',
                                borderRadius: '6px',
                                padding: '8px',
                                minWidth: '200px',
                                boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
                                zIndex: 999,
                                display: 'flex',
                                flexDirection: 'column',
                                gap: '4px',
                              }}
                            >
                              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px', marginBottom: '4px' }}>
                                <button 
                                  type="button"
                                  onClick={() => setSelectedWorkflows(allWorkflows)}
                                  style={{ border: 'none', background: 'none', color: 'var(--accent-highlight)', fontSize: '11px', cursor: 'pointer', fontWeight: 600 }}
                                >
                                  Select All
                                </button>
                                <button 
                                  type="button"
                                  onClick={() => setSelectedWorkflows([])}
                                  style={{ border: 'none', background: 'none', color: 'var(--text-secondary)', fontSize: '11px', cursor: 'pointer', fontWeight: 600 }}
                                >
                                  Clear
                                </button>
                              </div>
                              {allWorkflows.map(wf => {
                                const isChecked = selectedWorkflows.includes(wf);
                                return (
                                  <label 
                                    key={wf} 
                                    style={{
                                      display: 'flex',
                                      alignItems: 'center',
                                      gap: '8px',
                                      padding: '4px 6px',
                                      borderRadius: '4px',
                                      cursor: 'pointer',
                                      fontSize: '12px',
                                      transition: 'background 0.1s ease',
                                    }}
                                    onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'; }}
                                    onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent'; }}
                                  >
                                    <input 
                                      type="checkbox" 
                                      checked={isChecked}
                                      onChange={() => {
                                        if (isChecked) {
                                          setSelectedWorkflows(selectedWorkflows.filter(w => w !== wf));
                                        } else {
                                          setSelectedWorkflows([...selectedWorkflows, wf]);
                                        }
                                      }}
                                      style={{ cursor: 'pointer' }}
                                    />
                                    <span>{wf}</span>
                                  </label>
                                );
                              })}
                            </div>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Table Container */}
                    <div className="deepswe-container">
                      {/* Header */}
                      <div className="deepswe-header">
                        <div style={{ width: '210px', flexShrink: 0 }}>Workflow / Agent</div>
                        <div style={{ flex: 1 }}></div>
                        <div className="deepswe-desktop-stat deepswe-w-pass">Pass@1</div>
                        <div className="deepswe-desktop-stat deepswe-w-cost">Avg cost</div>
                        <div className="deepswe-desktop-stat deepswe-w-time">Avg time</div>
                        <div className="deepswe-desktop-stat deepswe-w-tok">Out tok</div>
                      </div>

                      {/* Rows */}
                      {displayedAccuracies.map((c) => {
                        const parsed = parseConditionName(c.condition);
                        const accuracyPercent = Math.round(c.planned_accuracy * 100);
                        const errMargin = Math.round(((c.planned_ci_upper - c.planned_ci_lower) / 2) * 100);
                        
                        const maxAccuracy = Math.max(...displayedAccuracies.map(acc => acc.planned_accuracy), 0.1);
                        let scaleMax = 1.0;
                        if (maxAccuracy <= 0.2) {
                          scaleMax = 0.2;
                        } else if (maxAccuracy <= 0.4) {
                          scaleMax = 0.4;
                        } else if (maxAccuracy <= 0.6) {
                          scaleMax = 0.6;
                        } else if (maxAccuracy <= 0.8) {
                          scaleMax = 0.8;
                        }

                        const barWidth = `${(c.planned_accuracy / scaleMax) * 100}%`;
                        const ciLeft = `${((c.planned_ci_lower || 0) / scaleMax) * 100}%`;
                        const ciWidth = `${(((c.planned_ci_upper || 0) - (c.planned_ci_lower || 0)) / scaleMax) * 100}%`;

                        const avgCost = c.expected > 0 ? c.cost / c.expected : 0;
                        const avgTime = c.avg_time;
                        const avgOutTok = c.expected > 0 ? c.output_tokens / c.expected : 0;

                        return (
                          <div key={c.condition} className="deepswe-row">
                            <div className="deepswe-model-col">
                              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0 }}>
                                <span style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: '18px', height: '18px', color: getWorkflowColor(parsed.name), flexShrink: 0 }}>
                                  {getWorkflowIcon(parsed.name)}
                                </span>
                                <span style={{ fontWeight: 500, fontSize: '13px', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }} title={parsed.name}>
                                  {parsed.name}
                                </span>
                                {parsed.effort && (
                                  <span style={{ color: 'var(--text-secondary)', fontSize: '11px', flexShrink: 0 }}>
                                    [{parsed.effort}]
                                  </span>
                                )}
                              </div>
                              
                              {/* Mobile pass@1 */}
                              <div className="sm-hidden-only">
                                <span style={{ fontSize: '13px', fontVariantNumeric: 'tabular-nums' }}>
                                  <span style={{ fontWeight: 600 }}>{accuracyPercent}%</span>
                                  <span style={{ fontSize: '11px', color: 'var(--text-secondary)', marginLeft: '4px' }}>±{errMargin}%</span>
                                </span>
                              </div>
                            </div>

                            {/* Bar Column */}
                            <div className="deepswe-bar-col">
                              <div style={{ position: 'relative', height: '20px' }}>
                                {/* Background track */}
                                <div style={{ position: 'absolute', top: '4px', bottom: '4px', left: 0, right: 0, backgroundColor: 'rgba(107, 114, 128, 0.08)', borderRadius: '3px' }} />
                                {/* Colored Bar */}
                                <div style={{ position: 'absolute', top: '4px', bottom: '4px', left: 0, width: barWidth, backgroundColor: getWorkflowColor(parsed.name), borderRadius: '3px', transition: 'width 0.3s ease' }} />
                                {/* Whisker line */}
                                <div style={{ position: 'absolute', top: '50%', transform: 'translateY(-50%)', left: ciLeft, width: ciWidth, height: '1px', backgroundColor: 'rgba(17, 24, 39, 0.65)' }} />
                                {/* Left whisker tick */}
                                <div style={{ position: 'absolute', top: '3px', bottom: '3px', left: ciLeft, width: '1px', backgroundColor: 'rgba(17, 24, 39, 0.65)' }} />
                                {/* Right whisker tick */}
                                <div style={{ position: 'absolute', top: '3px', bottom: '3px', left: `calc(${ciLeft} + ${ciWidth})`, width: '1px', backgroundColor: 'rgba(17, 24, 39, 0.65)' }} />
                              </div>
                            </div>

                            {/* Mobile Info Stats */}
                            <div className="deepswe-mobile-stats">
                              <span style={{ color: 'var(--text-secondary)' }}>Avg cost <span style={{ fontVariantNumeric: 'tabular-nums', color: 'var(--text-primary)', fontWeight: 500 }}>{formatAvgCost(avgCost)}</span></span>
                              <span style={{ color: 'var(--text-secondary)' }}>Avg time <span style={{ fontVariantNumeric: 'tabular-nums', color: 'var(--text-primary)', fontWeight: 500 }}>{formatAvgTime(avgTime)}</span></span>
                              <span style={{ color: 'var(--text-secondary)' }}>Out tok <span style={{ fontVariantNumeric: 'tabular-nums', color: 'var(--text-primary)', fontWeight: 500 }}>{formatAvgTokens(avgOutTok)}</span></span>
                            </div>

                            {/* Desktop Stats */}
                            <div className="deepswe-desktop-stat deepswe-w-pass">
                              <span style={{ fontSize: '13px' }}>
                                <span style={{ fontWeight: 600 }}>{accuracyPercent}%</span>
                                <span style={{ fontSize: '11px', color: 'var(--text-secondary)', marginLeft: '4px' }}>±{errMargin}%</span>
                              </span>
                            </div>
                            <div className="deepswe-desktop-stat deepswe-w-cost" style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{formatAvgCost(avgCost)}</div>
                            <div className="deepswe-desktop-stat deepswe-w-time" style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{formatAvgTime(avgTime)}</div>
                            <div className="deepswe-desktop-stat deepswe-w-tok" style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{formatAvgTokens(avgOutTok)}</div>
                          </div>
                        );
                      })}

                      {/* Bottom Ticks Axis */}
                      {displayedAccuracies.length > 0 && (() => {
                        const maxAccuracy = Math.max(...displayedAccuracies.map(acc => acc.planned_accuracy), 0.1);
                        let scaleMax = 1.0;
                        let ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
                        if (maxAccuracy <= 0.2) {
                          scaleMax = 0.2;
                          ticks = [0, 0.05, 0.1, 0.15, 0.2];
                        } else if (maxAccuracy <= 0.4) {
                          scaleMax = 0.4;
                          ticks = [0, 0.1, 0.2, 0.3, 0.4];
                        } else if (maxAccuracy <= 0.6) {
                          scaleMax = 0.6;
                          ticks = [0, 0.15, 0.3, 0.45, 0.6];
                        } else if (maxAccuracy <= 0.8) {
                          scaleMax = 0.8;
                          ticks = [0, 0.2, 0.4, 0.6, 0.8];
                        }

                        return (
                          <div className="deepswe-axis-container">
                            <div className="deepswe-model-col" style={{ height: 0, border: 'none', padding: 0 }} />
                            <div style={{ flex: 1, position: 'relative', height: '16px' }}>
                              {ticks.map((tickVal, idx) => {
                                const leftPercent = (tickVal / scaleMax) * 100;
                                let transform = 'translateX(-50%)';
                                if (idx === 0) transform = 'translateX(0)';
                                if (idx === ticks.length - 1) transform = 'translateX(-100%)';
                                return (
                                  <span 
                                    key={idx} 
                                    style={{
                                      position: 'absolute',
                                      left: `${leftPercent}%`,
                                      transform,
                                      fontSize: '10px',
                                      color: 'var(--text-muted)',
                                      fontVariantNumeric: 'tabular-nums',
                                    }}
                                  >
                                    {Math.round(tickVal * 100)}%
                                  </span>
                                );
                              })}
                            </div>
                            <div className="deepswe-desktop-stat deepswe-w-pass" />
                            <div className="deepswe-desktop-stat deepswe-w-cost" />
                            <div className="deepswe-desktop-stat deepswe-w-time" />
                            <div className="deepswe-desktop-stat deepswe-w-tok" />
                          </div>
                        );
                      })()}
                    </div>
                  </div>

                  {/* Paired Deltas Chart */}
                  <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '20px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <h3 style={{ fontSize: '13px', fontWeight: '700' }}>Paired Accuracy Difference</h3>
                      <div>
                        <span style={{ fontSize: '11px', color: 'var(--text-secondary)', marginRight: '6px' }}>Baseline:</span>
                        <select className="form-input" style={{ width: '130px', display: 'inline-block' }} value={baselineCondition} onChange={(e) => setBaselineCondition(e.target.value)}>
                          {details.summary.conditions.map(c => (
                            <option key={c.condition} value={c.condition}>{c.condition}</option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <div style={{ width: '100%', overflow: 'hidden' }}>
                      <svg viewBox="0 0 500 250" width="100%" height="250" style={{ background: 'white' }}>
                        {/* Axes */}
                        <line x1="120" y1="20" x2="120" y2="200" stroke="#000" />
                        {/* Zero difference line */}
                        <line x1="120" y1="110" x2="480" y2="110" stroke="#94a3b8" strokeDasharray="3,3" />

                        {/* Grid labels for differences (-100% to +100%) */}
                        {[-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6].map((val, idx) => {
                          const x = 300 + val * 300;
                          return (
                            <g key={idx}>
                              <line x1={x} y1="20" x2={x} y2="200" stroke="#f3f4f6" />
                              <text x={x} y="215" textAnchor="middle" style={{ fontSize: '9px' }}>
                                {val > 0 ? '+' : ''}{(val * 100).toFixed(0)}%
                              </text>
                            </g>
                          );
                        })}

                        {/* Render differences */}
                        {getDynamicPairedComparisons().map((d, idx, arr) => {
                          const h = 170 / arr.length;
                          const y = 30 + idx * h + h / 2;
                          const xCenter = 300 + d.delta * 300;
                          const xLower = 300 + (d.ci_lower || 0) * 300;
                          const xUpper = 300 + (d.ci_upper || 0) * 300;

                          return (
                            <g key={idx}>
                              <text x="110" y={y + 4} textAnchor="end" style={{ fontSize: '9px', fontWeight: '500' }}>
                                {d.condition.slice(0, 16)}
                              </text>

                              {/* Line for confidence interval */}
                              <line x1={xLower} y1={y} x2={xUpper} y2={y} stroke="#dc2626" strokeWidth="2" />
                              <line x1={xLower} y1={y - 4} x2={xLower} y2={y + 4} stroke="#dc2626" strokeWidth="2" />
                              <line x1={xUpper} y1={y - 4} x2={xUpper} y2={y + 4} stroke="#dc2626" strokeWidth="2" />

                              {/* Circle for point estimate */}
                              <circle cx={xCenter} cy={y} r="4.5" fill="#0f172a" />
                            </g>
                          );
                        })}
                      </svg>
                    </div>
                  </div>
                </div>

                {/* Cost/Efficiency Chart placed in Primary Results summary tab */}
                <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '24px', marginBottom: '24px', color: 'var(--text-primary)' }}>
                  <h3 style={{ fontSize: '14px', fontWeight: '700', marginBottom: '16px' }}>Cost and Efficiency Trade-off</h3>
                  {renderEfficiencyChart()}
                </div>

                {/* Paired Deltas table */}
                <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '20px' }}>
                  <h3 style={{ fontSize: '13px', fontWeight: '700', marginBottom: '12px' }}>Matched-task Paired Difference Ledger</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>Condition</th>
                        <th>Baseline</th>
                        <th style={{ textAlign: 'right' }}>Matched Pairs</th>
                        <th style={{ textAlign: 'right' }}>Acc Condition</th>
                        <th style={{ textAlign: 'right' }}>Acc Baseline</th>
                        <th style={{ textAlign: 'right' }}>Delta</th>
                        <th style={{ textAlign: 'right' }}>95% Clustered CI</th>
                      </tr>
                    </thead>
                    <tbody>
                      {getDynamicPairedComparisons().map((d, idx) => (
                        <tr key={idx}>
                          <td style={{ fontWeight: '600' }}>{d.condition}</td>
                          <td>{d.baseline}</td>
                          <td style={{ textAlign: 'right' }}>{d.matched_pairs}</td>
                          <td style={{ textAlign: 'right' }}>{(d.accuracy_cond * 100).toFixed(1)}%</td>
                          <td style={{ textAlign: 'right' }}>{(d.accuracy_baseline * 100).toFixed(1)}%</td>
                          <td style={{ textAlign: 'right', fontWeight: '600', color: d.delta > 0 ? 'var(--success-color)' : d.delta < 0 ? 'var(--error-color)' : 'inherit' }}>
                            {d.delta > 0 ? '+' : ''}{(d.delta * 100).toFixed(1)}%
                          </td>
                          <td style={{ textAlign: 'right', color: 'var(--text-secondary)' }}>
                            [{((d.ci_lower || 0) * 100).toFixed(1)}%, {((d.ci_upper || 0) * 100).toFixed(1)}%]
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* 3D. EFFICIENCY */}
            {activeTab === 'efficiency' && (
              <div>
                <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '24px', marginBottom: '24px', color: 'var(--text-primary)' }}>
                  {renderEfficiencyChart()}
                </div>

                {/* Tokens and cost table */}
                <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '20px' }}>
                  <h3 style={{ fontSize: '13px', fontWeight: '700', marginBottom: '12px' }}>Token and Efficiency Metrics</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>Condition</th>
                        <th style={{ textAlign: 'right' }}>Accuracy</th>
                        <th style={{ textAlign: 'right' }}>Cost</th>
                        <th style={{ textAlign: 'right' }}>Cost Per Correct</th>
                        <th style={{ textAlign: 'right' }}>Input Tokens</th>
                        <th style={{ textAlign: 'right' }}>Output Tokens</th>
                        <th style={{ textAlign: 'right' }}>Reasoning Tokens</th>
                        <th style={{ textAlign: 'right' }}>Total Tokens</th>
                        <th style={{ textAlign: 'right' }}>Correct / 1M Tokens</th>
                      </tr>
                    </thead>
                    <tbody>
                      {getFilteredAccuracies()
                        .filter(c => selectedGroups.includes(getGroupName(c)))
                        .map(c => {
                          const costPerCorrect = c.correct > 0 ? c.cost / c.correct : 0;
                          const correctPerMillion = c.tokens > 0 ? (c.correct / c.tokens) * 1000000 : 0;
                          
                          // Format label using same helper
                          const formatLabel = (id: string) => {
                            const match = id.match(/^([a-z-]+)-(?:e|w|s)-(.*)$/);
                            if (match) {
                              return `${match[1]} [${match[2]}]`;
                            }
                            return id;
                          };

                          return (
                            <tr 
                              key={c.condition}
                              onMouseEnter={() => setHoveredCondition(c.condition)}
                              onMouseLeave={() => setHoveredCondition(null)}
                              onClick={() => {
                                const gName = getGroupName(c);
                                if (selectedGroups.includes(gName)) {
                                  setSelectedGroups(selectedGroups.filter(sg => sg !== gName));
                                } else {
                                  setSelectedGroups([...selectedGroups, gName]);
                                }
                              }}
                              style={{
                                background: hoveredCondition === c.condition 
                                  ? 'var(--bg-tertiary)' 
                                  : 'transparent',
                                transition: 'background 0.15s ease',
                                cursor: 'pointer'
                              }}
                            >
                              <td style={{ fontWeight: '600', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <input 
                                  type="checkbox"
                                  checked={true}
                                  onChange={() => {}} // onClick on tr handles this
                                  style={{ cursor: 'pointer' }}
                                />
                                <span>{formatLabel(c.condition)}</span>
                              </td>
                              <td style={{ textAlign: 'right', fontWeight: '600' }}>{(c.planned_accuracy * 100).toFixed(1)}%</td>
                              <td style={{ textAlign: 'right' }}>{formatUSD(c.cost)}</td>
                              <td style={{ textAlign: 'right', fontWeight: '500' }}>
                                {costPerCorrect > 0 ? formatUSD(costPerCorrect) : 'n/a'}
                              </td>
                              <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{c.input_tokens.toLocaleString()}</td>
                              <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{c.output_tokens.toLocaleString()}</td>
                              <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{c.reasoning_tokens.toLocaleString()}</td>
                              <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{c.tokens.toLocaleString()}</td>
                              <td style={{ textAlign: 'right', fontWeight: '500' }}>
                                {correctPerMillion > 0 ? correctPerMillion.toFixed(1) : '0.0'}
                              </td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* 3E. PER-QUESTION MATRIX */}
            {activeTab === 'per_question' && (
              <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <h3 style={{ fontSize: '14px', fontWeight: '700' }}>Task-by-Condition Matrix</h3>
                  <div style={{ display: 'flex', gap: '12px', fontSize: '11px', color: 'var(--text-secondary)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <div style={{ width: '12px', height: '12px', background: '#16a34a', borderRadius: '2px' }}></div>
                      <span>Correct</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <div style={{ width: '12px', height: '12px', background: '#dc2626', borderRadius: '2px' }}></div>
                      <span>Incorrect</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <div style={{ width: '12px', height: '12px', background: '#ea580c', borderRadius: '2px' }}></div>
                      <span>Invalid</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <div style={{ width: '12px', height: '12px', background: '#9333ea', borderRadius: '2px' }}></div>
                      <span>Failed/Inconclusive</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <div style={{ width: '12px', height: '12px', background: '#e5e7eb', borderRadius: '2px' }}></div>
                      <span>Missing</span>
                    </div>
                  </div>
                </div>

                <div style={{ overflowX: 'auto' }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Task / Question Details</th>
                        <th>Type</th>
                        {details.summary.conditions.map(c => (
                          <th key={c.condition} style={{ width: '90px', fontSize: '10px' }}>{c.condition}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {details.tasks
                        .filter(t => (filterSubject === 'all' || t.category === filterSubject) && (filterAnswerType === 'all' || t.answer_type === filterAnswerType))
                        .map(task => (
                          <tr key={task.id}>
                            <td>
                              <div style={{ fontWeight: '600', maxWidth: '600px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                {task.prompt}
                              </div>
                              <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>
                                ID: {task.id} | Subject: {task.category}
                              </div>
                            </td>
                            <td>
                              <span style={{ fontSize: '10px', textTransform: 'capitalize' }}>
                                {task.answer_type.replace('_', ' ')}
                              </span>
                            </td>
                            {details.summary.conditions.map(cond => {
                              // Find run row matching task and condition
                              const runRow = details.runs.find(r => r.task_id === task.id && r.condition === cond.condition);
                              let cellBg = '#e5e7eb'; // missing fallback
                              let label = 'Missing';
                              
                              if (runRow) {
                                if (runRow.correct === true) {
                                  cellBg = '#dcfce7';
                                  label = 'Correct';
                                } else if (runRow.correct === false) {
                                  cellBg = '#fee2e2';
                                  label = 'Incorrect';
                                } else if (runRow.outcome === 'contract_invalid') {
                                  cellBg = '#ffedd5';
                                  label = 'Invalid';
                                } else if (runRow.outcome === 'inconclusive' || runRow.outcome === 'provider_execution_failure') {
                                  cellBg = '#f3e8ff';
                                  label = runRow.outcome.replace('_', ' ');
                                } else {
                                  cellBg = '#f3f4f6';
                                  label = 'Ungraded';
                                }
                              }

                              return (
                                <td key={cond.condition} style={{ padding: '6px' }}>
                                  <div
                                    style={{
                                      background: cellBg,
                                      border: '1px solid rgba(0,0,0,0.05)',
                                      borderRadius: '3px',
                                      padding: '8px 4px',
                                      textAlign: 'center',
                                      fontSize: '10px',
                                      fontWeight: '600',
                                      cursor: runRow ? 'pointer' : 'default',
                                      transition: 'transform 0.1s ease',
                                    }}
                                    onClick={() => runRow && handleOpenRunCellDetail(runRow)}
                                    title={runRow ? `Click to inspect: ${label}` : 'No run trace'}
                                  >
                                    {runRow ? runRow.outcome.slice(0, 10).replace('_', ' ') : 'missing'}
                                  </div>
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* 3F. MULTI-AGENT BEHAVIOR */}
            {activeTab === 'multi_agent' && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                {/* Transitions (Wrong-to-Right / Right-to-Wrong) */}
                <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '20px' }}>
                  <h3 style={{ fontSize: '13px', fontWeight: '700', marginBottom: '12px' }}>Revision Transitions (Stage-level Grading)</h3>
                  {details.summary.revision_transitions && details.summary.revision_transitions.length > 0 ? (
                    <table>
                      <thead>
                        <tr>
                          <th>Condition</th>
                          <th>Transition</th>
                          <th style={{ textAlign: 'right' }}>Occurrences</th>
                        </tr>
                      </thead>
                      <tbody>
                        {details.summary.revision_transitions.map((t, idx) => (
                          <tr key={idx}>
                            <td style={{ fontWeight: '600' }}>{t.condition}</td>
                            <td style={{ textTransform: 'capitalize' }}>{t.transition.replace('_', ' ')}</td>
                            <td style={{ textAlign: 'right', fontWeight: '500' }}>{t.count}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div style={{ padding: '24px', textAlign: 'center', color: 'var(--text-muted)' }}>
                      No stage-level grading data available. Stage responses must be graded to populate revision statistics.
                    </div>
                  )}
                </div>

                {/* Persuasion / Aggregation errors */}
                <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '20px' }}>
                  <h3 style={{ fontSize: '13px', fontWeight: '700', marginBottom: '12px' }}>Aggregation Error & Consensus</h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                      Below are condition error reasons tracked across runs. Focus on cases like <code>aggregation_error</code> (where a correct answer candidate was available but vote/judge chose incorrectly):
                    </div>
                    <table>
                      <thead>
                        <tr>
                          <th>Reason / Error Classification</th>
                          <th style={{ textAlign: 'right' }}>Count</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(details.summary.error_reasons || {}).map(([reason, count]) => (
                          <tr key={reason}>
                            <td style={{ fontFamily: 'var(--font-mono)', fontSize: '12px' }}>{reason}</td>
                            <td style={{ textAlign: 'right', fontWeight: '600', fontSize: '13px' }}>{count}</td>
                          </tr>
                        ))}
                        {Object.keys(details.summary.error_reasons || {}).length === 0 && (
                          <tr>
                            <td colSpan={2} style={{ textAlign: 'center', color: 'var(--text-muted)' }}>No execution errors or incorrect answers observed.</td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* 3G. CALIBRATION */}
            {activeTab === 'calibration' && (
              <div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '24px' }}>
                  {/* Reliability summary stats */}
                  <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '20px' }}>
                    <h3 style={{ fontSize: '13px', fontWeight: '700', marginBottom: '16px' }}>Calibration Statistics</h3>
                    <table>
                      <thead>
                        <tr>
                          <th>Condition</th>
                          <th style={{ textAlign: 'right' }}>Brier Score</th>
                          <th style={{ textAlign: 'right' }}>Expected Calibration Error (ECE)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {details.summary.conditions.map(c => (
                          <tr key={c.condition}>
                            <td style={{ fontWeight: '600' }}>{c.condition}</td>
                            <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                              {c.calibration ? c.calibration.brier_score.toFixed(4) : 'n/a'}
                            </td>
                            <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', fontWeight: '600' }}>
                              {c.calibration ? `${(c.calibration.expected_calibration_error * 100).toFixed(1)}%` : 'n/a'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Dynamic reliability bins */}
                  <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '20px' }}>
                    <h3 style={{ fontSize: '13px', fontWeight: '700', marginBottom: '12px' }}>Reliability Diagram (Count-Weighted Bins)</h3>
                    <div style={{ display: 'flex', gap: '12px' }}>
                      {details.summary.conditions.map(c => {
                        if (!c.calibration) return null;
                        return (
                          <div key={c.condition} style={{ flex: 1 }}>
                            <div style={{ fontSize: '11px', fontWeight: '600', textAlign: 'center', marginBottom: '6px' }}>{c.condition}</div>
                            <svg viewBox="0 0 150 150" width="100%" height="150" style={{ background: 'white', border: '1px solid #f3f4f6' }}>
                              {/* Diagonal perfection line */}
                              <line x1="15" y1="135" x2="135" y2="15" stroke="#94a3b8" strokeDasharray="2,2" />
                              
                              {/* Plot bin accuracies */}
                              {c.calibration.bins.map((b, idx) => {
                                if (b.count === 0) return null;
                                const x = 15 + b.avg_confidence * 120;
                                const y = 135 - b.accuracy * 120;
                                return (
                                  <g key={idx}>
                                    {/* Bar representing bin count weight */}
                                    <circle cx={x} cy={y} r={Math.min(10, Math.max(3, b.count * 1.5))} fill="#3b82f6" opacity="0.8" />
                                    <text x={x} y={y - 12} textAnchor="middle" style={{ fontSize: '7px' }}>n={b.count}</text>
                                  </g>
                                );
                              })}

                              {/* Labels */}
                              <text x="75" y="146" textAnchor="middle" style={{ fontSize: '8px' }}>Confidence</text>
                              <text x="6" y="75" textAnchor="middle" transform="rotate(-90 6 75)" style={{ fontSize: '8px' }}>Accuracy</text>
                            </svg>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 3H. JOB DETAILS */}
            {activeTab === 'runs' && (
              <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '20px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <h3 style={{ fontSize: '14px', fontWeight: '700' }}>Canonical Job Details</h3>
                  <input
                    type="text"
                    className="form-input"
                    style={{ width: '300px' }}
                    placeholder="Filter by task, condition, outcome, answer..."
                    value={runSearchQuery}
                    onChange={(e) => setRunSearchQuery(e.target.value)}
                  />
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Task ID</th>
                        <th>Repetition</th>
                        <th>Condition</th>
                        <th>Outcome</th>
                        <th style={{ textAlign: 'right' }}>Attempts</th>
                        <th>Score</th>
                        <th>Grading</th>
                        <th>Final Answer</th>
                        <th>Expected</th>
                        <th style={{ textAlign: 'right' }}>Tokens</th>
                        <th style={{ textAlign: 'right' }}>Cost</th>
                        <th style={{ width: '80px' }}>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {getFilteredRuns().map((run, idx) => (
                        <tr key={idx}>
                          <td style={{ fontFamily: 'var(--font-mono)', fontSize: '11px' }}>{run.task_id}</td>
                          <td style={{ textAlign: 'center' }}>{run.repetition}</td>
                          <td style={{ fontWeight: '500' }}>{run.condition}</td>
                          <td>
                            <span className={`outcome-${run.outcome}`}>
                              {run.outcome.replace('_', ' ')}
                            </span>
                          </td>
                          <td style={{ textAlign: 'right' }}>{run.attempt_count}</td>
                          <td>
                            {run.correct === true ? (
                              <span style={{ color: 'var(--success-color)', fontWeight: '600' }}>Correct</span>
                            ) : run.correct === false ? (
                              <span style={{ color: 'var(--error-color)', fontWeight: '600' }}>Incorrect</span>
                            ) : (
                              <span style={{ color: 'var(--text-muted)' }}>Ungraded</span>
                            )}
                          </td>
                          <td style={{ textTransform: 'capitalize' }}>{run.grading_status}</td>
                          <td style={{ maxWidth: '150px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={run.final_answer}>{run.final_answer}</td>
                          <td style={{ maxWidth: '150px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={run.expected_answer}>{run.expected_answer}</td>
                          <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{run.total_tokens.toLocaleString()}</td>
                          <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)' }}>{formatUSD(run.cost_usd)}</td>
                          <td>
                            <button className="btn btn-sm" onClick={() => handleOpenRunCellDetail(run)}>
                              Inspect
                            </button>
                          </td>
                        </tr>
                      ))}
                      {getFilteredRuns().length === 0 && (
                        <tr>
                          <td colSpan={12} style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
                            No runs match the filter criteria.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* 4. DETAIL RUN CELL MODAL */}
      {selectedCellRunId && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <div>
                <h2 style={{ fontSize: '14px', fontWeight: '700' }}>Run Trace: {selectedCellRunId}</h2>
                <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                  Condition: {selectedCellMetadata?.condition} | Task: {selectedCellMetadata?.task_id}
                </div>
              </div>
              <button className="btn btn-sm" onClick={() => setSelectedCellRunId(null)}>✕ Close</button>
            </div>

            <div className="modal-body">
              {traceLoading && (
                <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-secondary)' }}>
                  Loading execution trace and prompt variables...
                </div>
              )}

              {selectedCellTrace && !traceLoading && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                  {/* Left: Input, choices, gold, model output */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '16px' }}>
                      <h3 style={{ fontSize: '12px', fontWeight: '750', textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: '8px' }}>Task Question (Gold-Isolated)</h3>
                      <div style={{ fontSize: '13px', background: 'var(--bg-secondary)', padding: '10px', borderRadius: '4px', border: '1px solid var(--border-color)', whiteSpace: 'pre-wrap' }}>
                        {selectedCellTrace.task?.prompt || 'No question prompt stored'}
                      </div>
                      
                      {selectedCellTrace.task?.choices?.length > 0 && (
                        <div style={{ marginTop: '12px' }}>
                          <strong style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Options:</strong>
                          <ul style={{ listStyle: 'none', marginTop: '6px' }}>
                            {selectedCellTrace.task.choices.map((c: any) => (
                              <li key={c.label} style={{ padding: '4px 6px', background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: '2px', marginBottom: '4px', fontSize: '12px' }}>
                                <strong>{c.label}</strong>: {c.text}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>

                    <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '16px' }}>
                      <h3 style={{ fontSize: '12px', fontWeight: '750', textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: '8px' }}>Grader Verdict & Reference Answer</h3>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '12px' }}>
                        <div>
                          <strong style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Reference Answer:</strong>
                          <div style={{ fontSize: '13px', fontWeight: '600', color: 'var(--success-color)' }}>
                            {selectedCellMetadata?.expected_answer}
                          </div>
                        </div>
                        <div>
                          <strong style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Extracted Final Answer:</strong>
                          <div style={{ fontSize: '13px', fontWeight: '600' }}>
                            {selectedCellMetadata?.final_answer || 'n/a'}
                          </div>
                        </div>
                      </div>

                      <div style={{ background: 'var(--bg-secondary)', padding: '10px', borderRadius: '4px', border: '1px solid var(--border-color)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                          <span style={{ fontSize: '11px', fontWeight: '600' }}>Semantic Judge Reasoning:</span>
                          <span style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>
                            Confidence: {selectedCellMetadata?.grader_confidence || 'n/a'}%
                          </span>
                        </div>
                        <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                          {selectedCellMetadata?.grader_reasoning || 'No judge reasoning generated (exact matching fallback).'}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Right: Stages and Revisions */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '16px' }}>
                      <h3 style={{ fontSize: '12px', fontWeight: '750', textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: '12px' }}>Workflow Stage Revisions</h3>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                        {selectedCellTrace.stage_cards?.map((card: any, idx: number) => (
                          <div key={idx} style={{ border: '1px solid var(--border-color)', borderRadius: '4px', padding: '12px', background: 'var(--bg-secondary)' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                              <span style={{ fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', color: 'var(--text-secondary)' }}>
                                {card.step} (Agent: {card.agent_id})
                              </span>
                              <span style={{ fontSize: '10px', padding: '1px 5px', borderRadius: '2px', background: card.contract_valid ? '#dcfce7' : '#fee2e2', color: card.contract_valid ? '#15803d' : '#b91c1c' }}>
                                {card.contract_valid ? 'Valid Contract' : 'Invalid Contract'}
                              </span>
                            </div>
                            
                            <strong style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Answer:</strong>
                            <div style={{ fontSize: '12px', background: 'white', padding: '4px 6px', border: '1px solid var(--border-color)', borderRadius: '2px', marginTop: '2px', marginBottom: '8px' }}>
                              {card.answer || 'No answer extracted'}
                            </div>

                            <strong style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Raw Response / Thought Block:</strong>
                            <div style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', background: 'white', padding: '6px', border: '1px solid var(--border-color)', borderRadius: '2px', maxHeight: '120px', overflowY: 'auto', whiteSpace: 'pre-wrap', marginTop: '2px' }}>
                              {card.raw_response}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div style={{ background: 'var(--bg-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', padding: '16px' }}>
                      <h3 style={{ fontSize: '12px', fontWeight: '750', textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: '8px' }}>Metrics & Retries</h3>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                        <div>
                          <span style={{ fontSize: '10px', color: 'var(--text-secondary)', display: 'block' }}>Attempts / Retries:</span>
                          <span style={{ fontSize: '13px', fontWeight: '600' }}>{selectedCellMetadata?.attempt_count} attempts</span>
                        </div>
                        <div>
                          <span style={{ fontSize: '10px', color: 'var(--text-secondary)', display: 'block' }}>Total Spend (USD):</span>
                          <span style={{ fontSize: '13px', fontWeight: '600' }}>{formatUSD(selectedCellMetadata?.cost_usd || 0)}</span>
                        </div>
                        <div>
                          <span style={{ fontSize: '10px', color: 'var(--text-secondary)', display: 'block' }}>Wall Time Latency:</span>
                          <span style={{ fontSize: '13px', fontWeight: '600' }}>{((selectedCellMetadata?.wall_time_ms || 0) / 1000).toFixed(1)}s</span>
                        </div>
                        <div>
                          <span style={{ fontSize: '10px', color: 'var(--text-secondary)', display: 'block' }}>Tokens Used:</span>
                          <span style={{ fontSize: '12px', fontWeight: '600', fontFamily: 'var(--font-mono)' }}>
                            {selectedCellMetadata?.total_tokens.toLocaleString()}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="modal-footer">
              <button className="btn" onClick={() => setSelectedCellRunId(null)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
