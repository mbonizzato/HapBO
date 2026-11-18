TASK.md (below) describes a black-box optimization problem. Its domain description gives an approximate analytic
model of the objective (a formula with its constants). Your task is to write a PSL belief document whose compiled
preference map pi approximates that model as closely as you can over the whole search box: pi must be higher where the
model value is better (lower, since the objective is minimized).
This task replaces the instructions in TASK.md section 0 and its closing questions: use TASK.md only for the model,
the search space and the direction. Reproduce the model as given; do not widen, flatten or move features to allow
for its approximate calibration.

Three documents follow: TASK.md (the problem; the only source of information about the objective), PSL_VOCABULARY.md
(the PSL language, a closed vocabulary of belief tokens and how each one is compiled into pi; read it completely
before writing anything) and psl_schema.json (the JSON schema of a PSL document). A deterministic compiler turns the
document into pi exactly as PSL_VOCABULARY.md describes; you cannot run it here.

How the document will be judged: on a large uniform sample of points in the search box, pi is compared with the
model, by the rank correlation between pi and model goodness over the whole box, and by how well the best regions
agree (the overlap of the top 5% and of the top 10% of points ranked by pi and by the model).

How to work: you have no tools and cannot run code; work it out yourself, then write the document. The whole
PSL vocabulary is available: use whichever tokens and combinations approximate the model best.

Requirements for the document:
- it is valid against psl_schema.json and the rules in PSL_VOCABULARY.md, and uses only tokens defined there;
- its `schema.dimensions` list the parameters with exactly the names and in the order of TASK.md's search space;
- at most 60 tokens, counting every belief (including `baseline` and the meta tokens) and each anchor point as one;
- every belief that accepts a provenance uses `"provenance": "llm_domain"`, including `anchors` (where it is
  optional, include it); the trust block uses `"overall_confidence": "medium"`;
- all information comes from TASK.md only.

You may reason first. End your answer with the PSL document as a single JSON object inside one ```json code block;
only that block is read.

===== TASK.md =====
{TASK}

===== PSL_VOCABULARY.md =====
{VOCAB}

===== psl_schema.json =====
{SCHEMA}

===== end of documents =====
You may reason first. End your answer with the PSL document as a single JSON object inside one ```json code block;
only that block is read.
