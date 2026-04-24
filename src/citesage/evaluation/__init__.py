"""CiteSage evaluation package — Phase 3.

Entry point: ``python -m citesage.evaluation.run_eval``
"""

from .run_eval import EvalRunner, GoldenItem, QueryResult, load_dataset

__all__ = ["EvalRunner", "GoldenItem", "QueryResult", "load_dataset"]
