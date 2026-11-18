"""Worker-side entry points for the subprocess baseline bridge.

Each worker runs inside a baseline's isolated conda env, reads a problem spec on
stdin, drives the upstream optimizer over the finite candidate set, and requests
objective values from the parent via the JSON callback protocol in ``harness``.
"""
