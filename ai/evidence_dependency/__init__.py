"""Evidence-dependent delivery; one remote proposal, local execution and projection."""
from .protocol import build_prompt, generation_config, needs_calculation, needs_dependencies, parse_proposal, strategy
from .executor import execute
from .delivery import deliver
from .semantic import verify_relations


def execute_delivery(raw, candidates, question, *, logger=None, verifier=None):
    proposal = parse_proposal(raw)
    verifier = verifier or (lambda checks, sources, q: verify_relations(checks, sources, q, logger=logger))
    execution = execute(proposal, candidates, question, verifier)
    return deliver(execution, candidates, question), execution
