"""Elicit a PSL prior from the local LLM and commit it as a versioned artifact.

This is the offline elicitation step: it runs the LLM ONCE (before any BO loop),
validates + compiles the belief document, and writes a finalized, hash-provenanced
prior under ``experiments/priors/...`` that the study harness consumes unchanged.

Example (CNN leak-control task; HAPBO_VOCABULARY_CARD selects the ML-example vocabulary card):
    HAPBO_VOCABULARY_CARD=src/prior/expert_knowledge/vocabulary_card_ml_example.md \
    python scripts/elicit_prior.py \
        --dataset-id cnn-leak --prior-id my-prior \
        --domain-file leak_control_cnn/elicitation/cnn_leak_domain_overfit.txt \
        --optimization-file leak_control_cnn/elicitation/cnn_leak_opt.txt \
        --schema-json leak_control_cnn/elicitation/cnn_leak_schema.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))


def _text(inline: str | None, file: str | None) -> str:
    if file:
        return Path(file).read_text(encoding="utf-8")
    return inline or ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--prior-id", required=True)
    parser.add_argument("--target-name", default=None)
    parser.add_argument("--domain")
    parser.add_argument("--domain-file")
    parser.add_argument("--optimization")
    parser.add_argument("--optimization-file")
    parser.add_argument("--search-space")
    parser.add_argument("--search-space-file")
    parser.add_argument("--schema-json", help="JSON file with a dimensions list (or {schema:{dimensions}})")
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "openai"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", "none"))
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout", type=float, default=120.0,
                        help="per-request timeout in seconds (raise for Qwen thinking mode)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--no-structured",
        action="store_true",
        help="disable guided decoding; use free-form generation + validate-and-retry",
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--experiments-directory", default=None)
    parser.add_argument("--prompt-card-file", default=None,
                        help="send this complete card verbatim (instead of rendering domain/search/opt)")
    parser.add_argument("--card-human-statement", action="store_true",
                        help="the card carries a human (expert) statement: only then may beliefs be tagged "
                             "user_stated / user_confirmed (recorded as card.human_statement in metadata.json)")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")

    from prior.expert_knowledge.prompt_card import render_prompt_card, search_space_from_schema
    from prior.expert_knowledge.psl_schema import (
        VOCABULARY_CARD_PATH,
        assert_schema_matches_card,
        load_psl_schema,
    )
    from prior.landscape.landscape_generator import (
        finalize_prior_experiment,
        initialize_prior_experiment,
    )
    from llm.client import LLMClient
    from llm.psl_elicitor import elicit_belief_document

    assert_schema_matches_card()  # fail fast on schema drift

    domain = _text(args.domain, args.domain_file)
    optimization = _text(args.optimization, args.optimization_file)
    if args.schema_json:
        raw = json.loads(Path(args.schema_json).read_text(encoding="utf-8"))
        dimensions = raw if isinstance(raw, list) else raw.get("dimensions") or raw.get("schema", {}).get("dimensions", [])
        search_space = search_space_from_schema(dimensions)
    else:
        search_space = _text(args.search_space, args.search_space_file)
    if not (domain and optimization and search_space) and not args.prompt_card_file:
        parser.error("provide --domain, --optimization, and a search space (inline/file/schema)")

    init_kwargs: dict = {
        "dataset_id": args.dataset_id,
        "prior_id": args.prior_id,
        "provider": args.provider,
        "model": args.model,
        "card_human_statement": args.card_human_statement,
    }
    if args.target_name:
        init_kwargs["target_name"] = args.target_name
    if args.experiments_directory:
        init_kwargs["experiments_directory"] = args.experiments_directory
    prior_dir = initialize_prior_experiment(**init_kwargs)

    card_text = (
        Path(args.prompt_card_file).read_text(encoding="utf-8")
        if args.prompt_card_file
        else render_prompt_card(domain=domain, search_space=search_space, optimization_description=optimization)
    )
    (prior_dir / "prompt_card.md").write_text(card_text, encoding="utf-8", newline="\n")

    client = LLMClient(
        args.model,
        args.temperature,
        args.max_tokens,
        prior_dir / "llm_cache",
        args.budget,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        enable_thinking=bool(args.enable_thinking),
        top_p=args.top_p,
        request_timeout=args.request_timeout,
    )
    client.set_run_directory(prior_dir)

    result = elicit_belief_document(
        client,
        card_text,
        schema=load_psl_schema(),
        vocabulary_card_text=VOCABULARY_CARD_PATH.read_text(encoding="utf-8"),
        max_attempts=args.max_attempts,
        seed=args.seed,
        elicitation_replicate=args.replicate,
        use_structured_output=not args.no_structured,
        card_human_statement=args.card_human_statement,
    )

    (prior_dir / "prior.json").write_text(
        json.dumps(result.document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    (prior_dir / "raw_llm_response.txt").write_text(result.raw, encoding="utf-8", newline="\n")

    # Record generation provenance before finalizing (hashes the files).
    metadata_path = prior_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    from datetime import datetime, timezone

    metadata["generation"]["generated_at"] = datetime.now(timezone.utc).isoformat()
    metadata["generation"]["raw_llm_response_available"] = True
    metadata["generation"]["notes"] = (
        f"base_url={args.base_url}; seed={args.seed}; replicate={args.replicate}; "
        f"temperature={args.temperature}; enable_thinking={bool(args.enable_thinking)}; "
        f"card_file={bool(args.prompt_card_file)}; api_calls={client.call_count}"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    compiled = finalize_prior_experiment(prior_dir, generate_landscape=False)
    print(f"Elicited + finalized prior at {prior_dir}")
    print(f"  compiled alpha = {compiled.alpha:.3f}; uncached API calls = {client.call_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
