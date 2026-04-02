"""
Pytest collection guards.

Root-level `test_*.py` files here include interactive/utility scripts and should
not be imported as automated tests.
"""

collect_ignore = [
    "test_retrieval.py",
    "test_models.py",
]

