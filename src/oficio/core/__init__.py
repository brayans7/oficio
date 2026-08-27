"""Deterministic core. Pure: no I/O side effects beyond loading the price book, no LLM imports.

Hard rule: the agent layer imports core; core NEVER imports the agent layer.
"""
