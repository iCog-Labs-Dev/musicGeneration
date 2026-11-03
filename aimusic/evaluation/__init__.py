"""MUS-28 / REQ-18: reproducible long-form evaluation suite (Milestone 1).

Evaluation sits *downstream* of generation: it calls the public planning and
scoring APIs and never changes generation behavior.  Entry point::

    python -m aimusic.evaluation run --config config/benchmarks/m1_64_128.json --out outputs/benchmarks/m1

See ``docs/evaluation-suite.md``.
"""
