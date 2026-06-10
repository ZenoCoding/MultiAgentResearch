from multi_agent_research.workflows.base import Workflow
from multi_agent_research.workflows.debate import DebateWorkflow
from multi_agent_research.workflows.sample import IndependentSampleWorkflow
from multi_agent_research.workflows.self_critic import SelfCriticWorkflow
from multi_agent_research.workflows.solo import SoloWorkflow
from multi_agent_research.workflows.supervisor import SupervisorWorkflow

__all__ = [
    "DebateWorkflow",
    "IndependentSampleWorkflow",
    "SelfCriticWorkflow",
    "SoloWorkflow",
    "SupervisorWorkflow",
    "Workflow",
]
