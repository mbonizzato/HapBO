#!/usr/bin/env python3
"""Write every LLM-visible HPOBench task text from experiments/hpo_task_descriptors.json.

For each of the 40 tasks (5 model families x 8 datasets) this writes, under
experiments/priors/hpobench-<fam>-<task>/<task>/:

  base/prompt_card.md                 Domain + Search space + Optimization description
  inverted/prompt_card.md             byte copy of the base card (the inverted map lives in prior.json)
  llm-card/prompt_card.md             base card + open reasoning questions (section 4)
  llm-card-{pointonly,gaussian,directllm}/prompt_card.md
                                      byte copies of llm-card (the representation lives in prior.json)
  llm-card-misleading/prompt_card.md  llm-card + "## Domain guidance" + the task's misleading hint

and rewrites experiments/misleading_descriptions/hpobench-tabular/<fam>-<task>.txt (misleading
configuration kept, values rounded to 4 significant digits, one standard sentence). The search-space
section comes from the benchmark's grid definition (benchmark/HPOBench/<fam>/146818/search_space.json,
task-independent), so the benchmark data must be downloaded first (scripts/download_hpobench.py).

usage: python scripts/build_hpo_cards.py [--root <repo>] [--benchmark-dir benchmark/HPOBench]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prior.expert_knowledge.prompt_card import render_prompt_card  # noqa: E402

FAMS = ["lr", "rf", "xgb", "nn", "svm"]
BASE_SIBLINGS = ["inverted"]
CARD_SIBLINGS = ["llm-card-pointonly", "llm-card-gaussian", "llm-card-directllm"]

MODEL = {
    "lr": "a scikit-learn SGD-trained logistic-regression classifier (log-loss, L2 penalty)",
    "rf": "a scikit-learn random-forest classifier",
    "xgb": "an XGBoost gradient-boosted-tree classifier",
    "nn": "a scikit-learn multilayer-perceptron classifier (ReLU activations, Adam optimizer, "
          "a fixed number of training iterations)",
    "svm": "a scikit-learn SVM classifier with an RBF kernel",
}
KNOBS = {
    "lr": "The two knobs are the L2 regularization strength alpha and the initial SGD learning "
          "rate eta0, both on a log scale.",
    "rf": "The four knobs are max_depth (log), max_features (an exponent e: each split tries "
          "round(n_features ** e) features, so 0 means one feature and 1 means all features), "
          "min_samples_leaf, and min_samples_split (log).",
    "xgb": "The four knobs are the learning rate eta (log), max_depth (log), colsample_bytree "
           "(per-tree feature-subsampling fraction), and the L2 leaf regularization reg_lambda (log).",
    "nn": "The five knobs are network depth and width, the Adam initial learning rate, the L2 "
          "penalty alpha, and the mini-batch size.",
    "svm": "The two knobs are c, the SVM penalty parameter usually written C (larger c means "
           "weaker regularization), and gamma, the RBF kernel coefficient (larger gamma means a "
           "narrower kernel), both on a log scale.",
}
FAMILY_WORD = {"lr": "logistic-regression", "rf": "random-forest", "xgb": "gradient-boosted-tree",
               "nn": "multilayer-perceptron", "svm": "SVM"}

MEANING = {
    "alpha": {"lr": "L2 regularization strength", "nn": "L2 penalty on the weights"},
    "eta0": "initial SGD learning rate",
    "max_depth": "maximum tree depth",
    "max_features": "exponent e: each split tries round(n_features ** e) features (0 = one feature, 1 = all)",
    "min_samples_leaf": "minimum number of samples per leaf",
    "min_samples_split": "minimum number of samples required to split a node",
    "colsample_bytree": "per-tree feature-subsampling fraction",
    "eta": "boosting learning rate",
    "reg_lambda": "L2 leaf regularization",
    "batch_size": "samples per Adam update",
    "depth": "number of hidden layers, all of equal width",
    "learning_rate_init": "initial Adam step size",
    "width": "neurons per hidden layer",
    "c": "SVM penalty parameter (larger means weaker regularization)",
    "gamma": "RBF kernel coefficient (larger means a narrower kernel)",
}

# Section 4 of the llm-card: general-knowledge topics per family as open questions (no stated answers).
QUESTIONS = {
    "lr": ("How much regularization does logistic regression need on this kind of data? How "
           "does accuracy change at the largest and at the smallest alpha values, and does that "
           "depend on how clean or noisy the data are? How does accuracy change across the range "
           "of eta0, including its low and high ends?"),
    "rf": ("How sensitive is accuracy to each of the four knobs? How does accuracy change with "
           "max_depth from its smallest to its largest value? How does accuracy change as "
           "max_features goes from its smallest to its largest value? How do small versus large "
           "min_samples_leaf and min_samples_split affect the fit on tabular data?"),
    "xgb": ("How does accuracy change at low and at high eta? How does accuracy change across "
            "the range of max_depth? How does accuracy change across the range of "
            "colsample_bytree? How sensitive is accuracy to reg_lambda?"),
    "nn": ("How does accuracy change across the range of the Adam learning rate, including its "
           "extremes? What width and depth does a problem of this size need? How does the L2 "
           "penalty alpha affect accuracy on this kind of data? Does the class structure matter "
           "for any of these choices?"),
    "svm": ("Do c and gamma interact, and if so how? What shape would you expect the accuracy "
            "landscape to have over c and gamma? How does accuracy change at the extremes of c "
            "and of gamma, and at their combinations?"),
}


def search_space(fam: str, benchmark_dir: Path) -> str:
    """Family grid from the benchmark's own grid definition (task-independent).

    Every dimension is a finite ordinal grid; PSL forbids `scale` on ordinal dimensions,
    so log spacing is stated in the `meaning` field instead. SVM's native `C` is written
    `c` (PSL names are lowercase; the harness matches names case-insensitively)."""
    grid = json.loads((benchmark_dir / fam / "146818" / "search_space.json").read_text(encoding="utf-8"))
    blocks = []
    for i, d in enumerate(grid["dimensions"], start=1):
        name = d["name"].lower()
        meaning = MEANING[name][fam] if isinstance(MEANING[name], dict) else MEANING[name]
        if d.get("scale") == "log":
            meaning += "; values are log-spaced"
        values = [int(v) if float(v).is_integer() else v for v in d["values"]]
        blocks.append("\n".join([f"Parameter {i}", f"name: {name}", f"type: {d['type']}",
                                 f"values: {values}  # exact allowed values",
                                 f"meaning: {meaning}"]))
    return "\n\n".join(blocks)


def base_card(spec: dict, fam: str, task: str, space: str) -> str:
    t = spec["tasks"][task]
    phrase = f"{t['descriptor']} ({t['size']}, {t['features']}, {t['classes']})"
    domain = (f"System under study: hyperparameter optimization of {MODEL[fam]} on "
              f"{phrase}. {KNOBS[fam]} General {FAMILY_WORD[fam]} "
              f"hyperparameter-tuning knowledge applies.")
    optimization = (f"Objective: {spec['objective_text']}; higher is better.\n"
                    "Direction: maximize\n"
                    "Evaluation budget: about 8 to 32 evaluations (low-budget regime).\n"
                    "Known constraints or prior observations: none.")
    return render_prompt_card(domain=domain, search_space=space, optimization_description=optimization)


def sections(card: str):
    m = re.search(r"## 1\. Domain\n(.*?)\n## 2\. Search space\n(.*?)\n## 3\. Optimization description\n(.*)$",
                  card, re.S)
    if m is None:
        raise SystemExit("card not parsable")
    return [s.strip() for s in m.groups()]


def llm_card(fam: str, base: str) -> str:
    domain, space, opt = sections(base)
    text = render_prompt_card(domain=domain, search_space=space, optimization_description=opt)
    return (text + "\n## 4. Questions to consider before writing the prior\n" + QUESTIONS[fam]
            + f"\nUse general {FAMILY_WORD[fam]} hyperparameter-tuning knowledge to answer these for "
            "yourself, then write the prior.\n")


def misleading_path(mislead_dir: Path, fam: str, task: str) -> Path:
    path = mislead_dir / f"{fam}-{task.replace('-', '_')}.txt"
    return path if path.is_file() else mislead_dir / f"{fam}-{task}.txt"


def rewrite_misleading(path: Path, fam: str) -> str:
    """Normalize a misleading hint in place (configuration kept, values to 4 significant digits)."""
    text = path.read_text(encoding="utf-8").strip()
    m = re.search(r"near the configuration (.*?)\. Prefer", text)
    if not m:
        raise SystemExit(f"cannot parse misleading configuration in {path}")
    pairs = [p.strip().split("=") for p in m.group(1).split(",")]
    config = ", ".join(f"{k}={float(v):.4g}" for k, v in pairs)
    new = (f"Domain guidance for tuning this {FAMILY_WORD[fam]} classifier: "
           f"accuracy is MAXIMIZED near the configuration {config}. Prefer that region; "
           "configurations far from it underperform.")
    path.write_text(new + "\n", encoding="utf-8", newline="\n")
    return new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="repository root to write into (default: this repo)")
    ap.add_argument("--benchmark-dir", type=Path, default=None, help="default <root>/benchmark/HPOBench")
    args = ap.parse_args()
    root = args.root.resolve()
    benchmark_dir = args.benchmark_dir or root / "benchmark" / "HPOBench"
    spec = json.loads((root / "experiments" / "hpo_task_descriptors.json").read_text(encoding="utf-8"))
    priors = root / "experiments" / "priors"
    mislead_dir = root / "experiments" / "misleading_descriptions" / "hpobench-tabular"

    n = 0
    for fam in FAMS:
        space = search_space(fam, benchmark_dir)
        for task in spec["tasks"]:
            home = priors / f"hpobench-{fam}-{task}" / task
            base = base_card(spec, fam, task, space)
            card = llm_card(fam, base)
            hint = rewrite_misleading(misleading_path(mislead_dir, fam, task), fam)
            texts = {"base": base, "llm-card": card,
                     "llm-card-misleading": card.rstrip() + "\n\n## Domain guidance\n" + hint + "\n"}
            texts.update({pid: base for pid in BASE_SIBLINGS})
            texts.update({pid: card for pid in CARD_SIBLINGS})
            for prior_id, text in texts.items():
                d = home / prior_id
                d.mkdir(parents=True, exist_ok=True)
                (d / "prompt_card.md").write_text(text, encoding="utf-8", newline="\n")
                n += 1
    print(f"wrote {n} cards + {len(FAMS) * len(spec['tasks'])} misleading descriptions")


if __name__ == "__main__":
    main()
