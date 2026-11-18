#!/usr/bin/env python3
"""Agentic PSL elicitation on the synthetic panel: one coding-agent session per (elicitor, function).

Each session runs in a fresh workspace <work-dir>/<elicitor>/<fam>/ holding exactly
  TASK.md            experiments/synthetic_cards/rho90/<fam>/TASK.md
  PSL_VOCABULARY.md  experiments/synthetic_cards/PSL_VOCABULARY.md
  psl_schema.json    experiments/synthetic_cards/psl_schema.json
  psl_compiler.py    src/prior/landscape/landscape_generator.py (PROJECT_ROOT set to the workspace folder)
and receives the prompt experiments/synthetic_cards/AGENT_PROMPT_2N_<fam>.md as its only instruction. The prompt asks
for a document of at most 2 tokens per search-space parameter (2N: 12 / 14 / 20 / 30 tokens for Hartmann-6 /
Styblinski-Tang-7 / Levy-10 / Ackley-15), judged by rank correlation and top-5% / top-10% overlap with the card model.
The agent may write and run its own Python in the workspace; only prior.json is read.

Harnesses (--elicitor):
  claude-haiku-5-5, claude-sonnet-5-5, claude-opus-5-5, claude-fable-5-1
      Claude Code CLI: claude -p <prompt> --model <id> --permission-mode bypassPermissions
      --output-format stream-json --verbose, CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0 (the session
      waits for its own background jobs), wall-clock cap --timeout (default 3600 s). Token totals and the list-price
      cost come from the stream's final result event.
  qwen
      Qwen Code CLI: qwen <prompt> --yolo --auth-type openai --openai-base-url $LLM_BASE_URL -m $LLM_MODEL
      -o stream-json, with a fresh HOME holding only .qwen/settings.json (telemetry off; memory, web, sub-agent, skill,
      monitor, scheduling, LSP, user-question and MCP-resource tools disabled), cap --timeout. With --bwrap the session
      runs in a bubblewrap sandbox in which only the workspace (rw), the per-run HOME (rw), the Python prefix and the
      node prefix (ro), /usr and /etc (ro) are visible. Cost 0 (local endpoint).

A session counts as done when the CLI exits with code 0, prior.json exists and the stream ends without an API error;
a done marker makes later calls skip it. The document is copied to <out-root>/<elicitor>/<fam>/prior.json and the token /
cost totals to usage.json next to it. The stream log and stderr stay under <work-dir>/_logs/.

usage: python scripts/agentic_elicit.py --elicitor claude-opus-5-5 --work-dir <dir> [--fams hartmann6,...]
       python scripts/agentic_elicit.py --elicitor qwen --work-dir <dir> [--bwrap --python-prefix P --node-prefix N]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "experiments" / "synthetic_cards"
COMPILER = ROOT / "src" / "prior" / "landscape" / "landscape_generator.py"
FAMS = ["hartmann6", "styblinski_tang7", "levy10", "ackley15"]
CLAUDE_MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"]
QWEN_DISABLED = ["save_memory", "web_fetch", "web_search", "agent", "skill", "monitor", "cron_create", "cron_list",
                 "cron_delete", "lsp", "ask_user_question", "loop_wakeup", "read_mcp_resource"]


# ------------------------------------------------------------------------------------------------ workspace
def build_workspace(ws: Path, fam: str) -> None:
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    shutil.copyfile(CARDS / "rho90" / fam / "TASK.md", ws / "TASK.md")
    shutil.copyfile(CARDS / "PSL_VOCABULARY.md", ws / "PSL_VOCABULARY.md")
    shutil.copyfile(CARDS / "psl_schema.json", ws / "psl_schema.json")
    src = COMPILER.read_text(encoding="utf-8")
    line = "PROJECT_ROOT = Path(__file__).resolve().parents[3]\n"
    if line not in src:
        raise RuntimeError("compiler PROJECT_ROOT line not found")
    src = src.replace(line, "PROJECT_ROOT = Path(__file__).resolve().parent\n")
    (ws / "psl_compiler.py").write_bytes(src.encode("utf-8"))


# ------------------------------------------------------------------------------------------------ stream parsing
def _events(path: Path) -> list[dict]:
    out = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


def _fallback(ev: list[dict]) -> dict:
    """Sum of the last streamed usage per assistant message (used when the result event is missing)."""
    msg = {}
    for e in ev:
        m = e.get("message", {}) if e.get("type") == "assistant" else {}
        if m.get("usage"):
            msg[m.get("id")] = m["usage"]
    return {"input": sum(u.get("input_tokens", 0) for u in msg.values()),
            "cache_read": sum(u.get("cache_read_input_tokens", 0) or 0 for u in msg.values()),
            "cache_creation": sum(u.get("cache_creation_input_tokens", 0) or 0 for u in msg.values()),
            "output": sum(u.get("output_tokens", 0) for u in msg.values())}


def claude_usage(log: Path) -> dict:
    """Token totals from the final result event (modelUsage summed over models) and its total_cost_usd."""
    ev = _events(log)
    res = [e for e in ev if e.get("type") == "result"]
    if not res:
        return {"complete": False, "cost_usd": None, **_fallback(ev)}
    r = res[-1]
    mu, u = r.get("modelUsage") or {}, r.get("usage") or {}
    return {"complete": True, "num_turns": r.get("num_turns"), "wall_s": round(r.get("duration_ms", 0) / 1000, 1),
            "cost_usd": r.get("total_cost_usd"),
            "input": sum(v.get("inputTokens", 0) for v in mu.values()) or u.get("input_tokens", 0),
            "cache_read": sum(v.get("cacheReadInputTokens", 0) for v in mu.values())
            or u.get("cache_read_input_tokens", 0),
            "cache_creation": sum(v.get("cacheCreationInputTokens", 0) for v in mu.values())
            or u.get("cache_creation_input_tokens", 0),
            "output": sum(v.get("outputTokens", 0) for v in mu.values()) or u.get("output_tokens", 0)}


def qwen_usage(log: Path) -> dict:
    """Token totals from the final result event's usage (no cache creation; reasoning is inside output)."""
    ev = _events(log)
    res = [e for e in ev if e.get("type") == "result"]
    if not res:
        fb = _fallback(ev)
        fb["cache_creation"] = 0
        return {"complete": False, "cost_usd": 0.0, **fb}
    r = res[-1]
    u = r.get("usage") or {}
    return {"complete": True, "num_turns": r.get("num_turns"), "wall_s": round(r.get("duration_ms", 0) / 1000, 1),
            "cost_usd": 0.0, "input": u.get("input_tokens", 0), "cache_read": u.get("cache_read_input_tokens", 0) or 0,
            "cache_creation": 0, "output": u.get("output_tokens", 0)}


def total_tokens(u: dict) -> int:
    return int(sum(u.get(k, 0) or 0 for k in ("input", "cache_read", "cache_creation", "output")))


# ------------------------------------------------------------------------------------------------ sessions
def claude_command(prompt: str, model: str) -> tuple[list[str], dict]:
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN") and not k.startswith("CLAUDE")}
    env["CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS"] = "0"
    cmd = [os.environ.get("CLAUDE_CLI", "claude"), "-p", prompt, "--model", model,
           "--permission-mode", "bypassPermissions", "--output-format", "stream-json", "--verbose"]
    return cmd, env


def qwen_command(prompt: str, ws: Path, home: Path, a) -> tuple[list[str], dict, Path | None]:
    url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
    model = os.environ.get("LLM_MODEL", "Qwen3.5-122B-A10B-GPTQ-Int4")
    key = os.environ.get("LLM_API_KEY", "none")
    (home / ".qwen").mkdir(parents=True, exist_ok=True)
    settings = {"model": {"name": model}, "baseUrl": url, "apiKey": key, "authType": "openai",
                "telemetry": {"enabled": False}, "tools": {"disabled": QWEN_DISABLED},
                "permissions": {"deny": QWEN_DISABLED}, "$version": 4}
    (home / ".qwen" / "settings.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")
    qwen = [os.environ.get("QWEN_CLI", "qwen"), prompt, "--yolo", "--auth-type", "openai", "--openai-base-url", url,
            "-m", model, "-o", "stream-json"]
    if not a.bwrap:
        env = dict(os.environ, HOME=str(home), OPENAI_API_KEY=key, QWEN_CODE_SUPPRESS_YOLO_WARNING="1")
        return qwen, env, ws
    if not (a.python_prefix and a.node_prefix):
        raise SystemExit("--bwrap needs --python-prefix and --node-prefix")
    py, node = str(Path(a.python_prefix).resolve()), str(Path(a.node_prefix).resolve())
    cmd = ["bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--symlink", "usr/sbin", "/sbin", "--ro-bind", "/etc", "/etc",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/home",
           "--ro-bind", node, node, "--ro-bind", py, py, "--bind", str(home), "/agent-home", "--bind", str(ws), "/work",
           "--chdir", "/work", "--clearenv", "--setenv", "LANG", "C.UTF-8", "--setenv", "HOME", "/agent-home",
           "--setenv", "PATH", f"{py}/bin:{node}/bin:/usr/bin:/bin", "--setenv", "OPENAI_API_KEY", key,
           "--setenv", "QWEN_CODE_SUPPRESS_YOLO_WARNING", "1", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
           "--die-with-parent", *qwen]
    return cmd, dict(os.environ), None


def run_one(elicitor: str, fam: str, a) -> None:
    work = Path(a.work_dir).resolve()
    logs = work / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    rid = f"{elicitor}__{fam}"
    if (logs / f"{rid}.done").exists():
        print(f"skip {rid} (done)")
        return
    ws = work / elicitor / fam
    build_workspace(ws, fam)
    prompt = (CARDS / f"AGENT_PROMPT_2N_{fam}.md").read_text(encoding="utf-8")
    if elicitor == "qwen":
        home = work / "_home" / rid
        if home.exists():
            shutil.rmtree(home)
        cmd, env, cwd = qwen_command(prompt, ws, home, a)
    else:
        (cmd, env), cwd = claude_command(prompt, elicitor), ws
    t0 = time.time()
    with open(logs / f"{rid}.jsonl", "wb") as out, open(logs / f"{rid}.err", "wb") as err:
        try:
            rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                timeout=a.timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    seconds = int(time.time() - t0)
    with open(logs / f"{rid}.err", "a", encoding="utf-8") as err:
        err.write(f"exit={rc} seconds={seconds}\n")
    tail = (logs / f"{rid}.jsonl").read_bytes()[-2000:].decode("utf-8", errors="replace")
    done = rc == 0 and (ws / "prior.json").is_file() and '"api_error":' not in tail
    usage = (qwen_usage if elicitor == "qwen" else claude_usage)(logs / f"{rid}.jsonl")
    usage.update(elicitor=elicitor, family=fam, exit=rc, wall_s_cli=seconds, done=done,
                 total_tokens=total_tokens(usage))
    if (ws / "prior.json").is_file():
        dst = Path(a.out_root).resolve() / elicitor / fam
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ws / "prior.json", dst / "prior.json")
        (dst / "usage.json").write_text(json.dumps(usage, indent=1), encoding="utf-8")
    if done:
        (logs / f"{rid}.done").touch()
    print(f"{rid}: exit={rc} seconds={seconds} prior={(ws / 'prior.json').is_file()} done={done} "
          f"tokens={usage['total_tokens']} cost={usage.get('cost_usd')}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--elicitor", required=True, choices=["qwen", *CLAUDE_MODELS])
    ap.add_argument("--work-dir", required=True, help="workspaces and logs (outside any folder with agent settings)")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "elicitation" / "agentic"))
    ap.add_argument("--fams", default=",".join(FAMS))
    ap.add_argument("--timeout", type=int, default=3600, help="wall-clock cap per session, seconds")
    ap.add_argument("--bwrap", action="store_true", help="qwen: run inside a bubblewrap sandbox (Linux)")
    ap.add_argument("--python-prefix", help="qwen --bwrap: Python environment prefix mounted read-only")
    ap.add_argument("--node-prefix", help="qwen --bwrap: node installation prefix mounted read-only")
    a = ap.parse_args()
    for fam in a.fams.split(","):
        if fam not in FAMS:
            raise SystemExit(f"unknown family {fam!r}")
        run_one(a.elicitor, fam, a)


if __name__ == "__main__":
    sys.exit(main())
