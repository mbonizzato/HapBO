"""Verify that the Anthropic API key (ANTHROPIC_API_KEY, environment or a local .env) works."""

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL = "claude-sonnet-5-5"


def main() -> None:
    """Read the API key and make a small authenticated API request."""

    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        load_dotenv(env_file)  # values already in the environment take precedence
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set (environment or .env)")

    client = anthropic.Anthropic()  # Reads ANTHROPIC_API_KEY from the environment.
    model = client.models.retrieve(MODEL)

    print("Anthropic API key is working.")
    print(f"API access confirmed; selected model: {model.id}")


if __name__ == "__main__":
    main()
