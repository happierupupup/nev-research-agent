"""S2 role packages: Supervisor / Research / Analyst / Verifier."""

from agent.agents.analyst import run_node as analyst_node
from agent.agents.research import run_node as research_node
from agent.agents.supervisor import decide_node as supervisor_decide_node
from agent.agents.supervisor import plan_node as supervisor_plan_node
from agent.agents.verifier import run_node as verifier_node

__all__ = [
    "analyst_node",
    "research_node",
    "supervisor_decide_node",
    "supervisor_plan_node",
    "verifier_node",
]
