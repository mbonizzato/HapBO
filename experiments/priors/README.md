# Versioned expert priors

Each leaf directory under this folder is one immutable, named prior used as an
input to Bayesian optimization. Organize new priors as:

```text
experiments/priors/<dataset-id>/<target-slug>/<prior-id>/
├── prompt_card.md
├── prior.json
├── raw_llm_response.txt       # written by scripts/elicit_prior.py; gitignored, not shipped
└── metadata.json
```

Use a new `<prior-id>` whenever the prompt, vocabulary, model output, or manual
correction changes. Do not overwrite a prior after it has been used for a
reported experiment. The small text artifacts in this directory are tracked in
version control.

Create a target-aware draft with `python main.py prior-init --dataset-id
<dataset-id> --prior-target-name <exact-target-name> --prior-id <prior-id>`.
For NeuroMOSAIC, pass the exact case-sensitive target with `--muscle` instead;
that argument is required for `prior-init`. The command derives `<target-slug>`,
copies the reusable prompt template, and records both the exact target and slug
in draft metadata. Fill the prompt card and save the validated JSON as
`prior.json`; a raw model response saved beside it stays local (gitignored).

For example, pass `--muscle LeftTibialisAnterioris`, which records
`scope.target_name` as `LeftTibialisAnterioris` and derives
`scope.target_slug` as `left-tibialis-anterioris`. Never pass the slug itself as
the muscle name. Keep `prior.json` as a separate file beside `metadata.json`;
finalization records its SHA-256 hash rather than embedding its contents.

At runtime, a versioned NeuroMOSAIC prior is rejected unless its metadata
dataset and exact target name match the selected dataset and muscle. Two-level
prior directories are readable for non-NeuroMOSAIC workflows, but
cannot satisfy this exact-target compatibility check.

HapBO resolves a prior automatically from this layout. When a target is
selected, it searches
`<dataset-id>/<target-slug>/<prior-id>/prior.json`; benchmark instance IDs such
as HPOBench and SOLAR use `<dataset-id>/<prior-id>/prior.json`. If exactly one
version exists it is selected automatically. Use `--prior-id` when multiple
versions exist, or `--dataset-prior-file` to override convention-based lookup.
If no matching `prior.json` exists, the command reports the expected location.

The vocabulary is not duplicated. Every prior references the vocabulary card it
was elicited with by repository path and SHA-256 hash:
`src/prior/expert_knowledge/vocabulary_card.md` (default card),
`vocabulary_card_neutral_example.md` (cortical priors `llm-geometry`,
`llm-expert-hint` and `human-expert-map-shared`) or
`vocabulary_card_ml_example.md` (CNN leak-control prior). The environment
variable `HAPBO_VOCABULARY_CARD` (path relative to the repository root) selects
the card of a new elicitation. Run `python main.py prior-finalize --prior-directory <directory>` after
the prompt and prior are complete. Finalization validates the prior, computes
all hashes, locks the metadata, and generates the landscape diagnostics.

Provenance `user_stated` / `user_confirmed` names a human source. A prior may use
it only when its prompt card carries a human (expert) statement, declared as
`card.human_statement: true` in `metadata.json` (`main.py prior-init
--card-human-statement`, `scripts/elicit_prior.py --card-human-statement`).
Loading and finalization reject a `user_stated` / `user_confirmed` tag in a
prior whose card has no human-statement declaration, and the elicitor's
validate-and-retry loop returns it to the model as a validator error.

The vocabulary card describes the meta tokens (`smoothness`, `noise`,
`modality`) as optimizer hints; this implementation validates them but does not
use them.

Generated plots, diagnostic samples, metrics, and optimization traces belong
under `results`, grouped by the same dataset, target, and prior IDs.

## Shipped priors

| Directory | Prior ids |
| --- | --- |
| `hpobench-<model>-<task>/<task>/` | `base` (base card), `llm-card` (task card), `llm-card-misleading` (card plus misleading hint), `llm-card-{pointonly,gaussian,directllm}`, `llm-card-misleading-{pointonly,gaussian,directllm}` (representation variants), `inverted` (inverted map), `caliper-card`, `caliper-card-misleading` (CALIPER multi-role priors, `caliper_prior.json.gz`), `neps-point-card`, `neps-point-card-misleading` (PriorBand / PriMO point priors, `neps_prior.json`), `llm-ei-card`, `llm-ei-card-misleading` (LLM-EI survey answers `survey_answers.json` and utilities `llm_ei_utility.json.gz`) |
| `neuromosaic-dataset1-rat/<muscle>/` | `human-expert-map` (human-made expert map), `human-expert-map-shared` (shared text of the baselines), `llm-expert-hint`; `llm-geometry` for the three muscles of the flexor benchmark (`left-knee-flexor`, `left-tibialis-anterioris`, `right-tibialis-anterioris`) |
| `solar-maxeff-re/` | `expert` (SOLAR card prior) |
| `cnn-leak/` | `card-overfit` (CNN leak-control prior) |

The PSL documents of the elicitor comparison (fig_elicitation_ladder) are in `experiments/psl_elicitor_priors/<elicitor>/{draw1,draw2,draw3,agentic}/<function>/prior.json` (rho90 cards; one API call per draw, or one agentic session).

Raw model responses and elicitation logs are not shipped. The compiled synthetic-panel documents are
in `experiments/synthetic_priors/`, the LLM inputs in `experiments/synthetic_cards/`, and the
ColaBO (c, s) beliefs in `experiments/colabo_llm_priors/`.

The headers `# Prompt A -- LLM as an expert.` and `# Prompt B` of the committed prompt files name the
prompt: Prompt A is the HapBO prompt card for PSL elicitation
(`src/prior/expert_knowledge/prompt_card_template.md`, rendered by
`src/prior/expert_knowledge/prompt_card.py`); Prompt B is the point-prior prompt of the ColaBO and HypBO
baselines (`*.prompt.md` in `experiments/colabo_llm_priors/` and `experiments/hypbo_llm_hypotheses/`), the
same card (cortical: the shared text of the condition) with its PSL instruction replaced by a request for
one (c, s) guess (ColaBO: `scripts/elicit_colabo_llm_priors.py`, `scripts/elicit_colabo_llm.py`) or a list
of (c, s) hypotheses (HypBO: `scripts/elicit_hypbo_llm.py`).
