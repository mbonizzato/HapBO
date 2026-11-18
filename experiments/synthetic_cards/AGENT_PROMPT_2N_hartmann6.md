You are working inside the current folder. Only use and produce files in this folder.

TASK.md describes a black-box optimization problem. Its domain description gives an approximate analytic model of the
objective (a formula with its constants). Your task is to write a PSL belief document, saved as `prior.json` in this
folder, whose compiled preference map pi approximates that model as closely as you can over the whole search box:
pi must be higher where the model value is better (lower, since the objective is minimized).
This task replaces the instructions in TASK.md section 0 and its closing questions: use TASK.md only for the model,
the search space and the direction. Reproduce the model as given; do not widen, flatten or move features to allow
for its approximate calibration.

Files:
- `TASK.md`: the problem; the only source of information about the objective.
- `PSL_VOCABULARY.md`: the PSL language, a closed vocabulary of belief tokens and how each one is compiled into pi.
  Read it completely before writing anything.
- `psl_schema.json`: the JSON schema of a PSL document.
- `psl_compiler.py`: the deterministic compiler. In Python: `from psl_compiler import validate_record, compile_prior`;
  `validate_record(doc)` raises if the document is invalid; `compile_prior(doc).pi(X)` returns pi at points X given in
  the native units of the search space (an array of shape (n_points, n_dims), columns in the order of your document's
  `schema.dimensions`, which must list the parameters with exactly the names and in the order of TASK.md's search
  space). Use only these two functions; do not run psl_compiler.py as a script and do not modify it.

How the document will be judged: on a large uniform sample of points in the search box, pi is compared with the
model, by the rank correlation between pi and model goodness over the whole box, and by how well the best regions
agree (the overlap of the top 5% and of the top 10% of points ranked by pi and by the model). Only `prior.json` is
used; nothing else you write is read.

How to work: the approach is up to you. Write and run your own Python (numpy and scipy are available) to analyse the
model, to compile and evaluate candidate documents, and to fit or search over the tokens and the numbers inside them.
The whole PSL vocabulary is available: use whichever tokens and combinations approximate the model best. Evaluate
your candidates the way they will be judged, keep improving while you can, and stop when you can no longer improve
the match. Save your best valid `prior.json` whenever it improves, so that a valid document always exists.

Requirements for the final `prior.json`:
- it passes `validate_record` and uses only tokens defined in PSL_VOCABULARY.md;
- at most 12 tokens (2 per search-space parameter; this problem has 6 parameters), counting every belief (including `baseline` and the meta tokens) and each anchor point as one;
- every belief that accepts a provenance uses `"provenance": "llm_domain"`, including `anchors` (where it is
  optional, include it); the trust block uses `"overall_confidence": "medium"`;
- all information comes from TASK.md only.
