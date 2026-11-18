"""LGBO (LLM-guided BO with a region-lifted posterior) worker for a finite candidate set.

From-paper implementation of the LGBO update. Each BO iteration the LLM returns a point
or a region with a confidence ``c``; the region is discretized on a Sobol grid ``G`` with
weights ``a``, and the current GP posterior mean is shifted by

    c * Sigma(X, G) a / sqrt(a.T Sigma(G, G) a),

where every Sigma is the current posterior covariance. The suggested regional functional
therefore moves by exactly ``c`` posterior standard deviations; the covariance is
unchanged and zero posterior variance disables the lift. Log-EI on the shifted posterior
selects the next finite candidate. The GP is a Matern-5/2 with per-iteration
marginal-likelihood refit; two LLM-chosen points form the initial design.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lgbo_common as common  # noqa: E402
from finite import normalize_rows, seeded_initial_indices  # noqa: E402
from harness import finish, log, make_evaluator, read_spec  # noqa: E402


# --------------------------------------------------------------------------- #
# LLM telemetry: tokens + wall-time per call. The instrumented chat sends the standard
# request (same payload, seed policy, thinking disabled) and records, for every call, the
# prompt/completion/total tokens the server reports and the measured latency. On finish()
# the telemetry is written to the path in ``LGBO_TELEMETRY`` (set by the parent per run);
# without the env var it is a no-op.
# --------------------------------------------------------------------------- #
_TELEMETRY: list[dict] = []


def _make_instrumented_chat(*, base_url, api_key, model, timeout, seed, sink):
    """Return a synchronous ``chat(messages, **kw) -> str`` that records token usage and latency in ``sink``."""

    import aiohttp  # lazily, so importing this module never requires aiohttp

    endpoint = base_url.rstrip("/") + "/chat/completions"
    request_ids = itertools.count()

    async def one_completion(payload):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer %s" % api_key
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout, trust_env=True) as session:
            async with session.post(endpoint, headers=headers, json=payload) as response:
                text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(
                        "LGBO local LLM returned HTTP %d: %s" % (response.status, text[:1000])
                    )
                try:
                    return await response.json()
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        "LGBO local LLM returned invalid JSON: %s" % text[:1000]
                    ) from exc

    def chat(messages, temperature=0.0, max_tokens=512, kind="guidance", query_after=None):
        ordinal = next(request_ids)
        payload = {
            "model": model,
            "messages": [dict(m) for m in messages],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": 0.95,
            "n": 1,
            "seed": int(seed + ordinal * 10_000),
            "chat_template_kwargs": {"enable_thinking": False},
        }
        started = time.perf_counter()
        ok = True
        usage = {}
        try:
            response = asyncio.get_event_loop().run_until_complete(one_completion(payload))
            choices = response.get("choices") or []
            if not choices:
                raise RuntimeError("LGBO local LLM response contained no choices")
            usage = response.get("usage") or {}
            content = str((choices[0].get("message") or {}).get("content") or "")
        except Exception:
            ok = False
            content = ""
            raise
        finally:
            sink.append({
                "kind": str(kind),
                "query_after": None if query_after is None else int(query_after),
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
                "latency_seconds": time.perf_counter() - started,
                "ok": ok,
            })
        return content

    return chat


def _write_telemetry(config, budget, dim, seed):
    """Serialize accumulated LLM telemetry to ``LGBO_TELEMETRY`` (if set)."""

    path = os.environ.get("LGBO_TELEMETRY")
    if not path:
        return
    calls = list(_TELEMETRY)
    guidance = [c for c in calls if c["kind"] == "guidance"]
    n_ok = sum(1 for c in calls if c["ok"])

    def _sum(key, rows):
        return int(sum(c[key] for c in rows))

    def _mean(key, rows):
        return (float(sum(c[key] for c in rows)) / len(rows)) if rows else 0.0

    summary = {
        "model": str(config.get("llm_model") or os.environ.get("LLM_MODEL") or "local-model"),
        "budget": int(budget),
        "dim": int(dim),
        "seed": int(seed),
        "n_calls": len(calls),
        "n_calls_ok": n_ok,
        "n_calls_failed": len(calls) - n_ok,
        "n_initial_calls": sum(1 for c in calls if c["kind"] == "initial"),
        "n_guidance_calls": len(guidance),
        "prompt_tokens_total": _sum("prompt_tokens", calls),
        "completion_tokens_total": _sum("completion_tokens", calls),
        "total_tokens_total": _sum("total_tokens", calls),
        "llm_latency_seconds_total": float(sum(c["latency_seconds"] for c in calls)),
        # Per-query = per guidance call (one LLM query per BO iteration after init).
        "prompt_tokens_per_query_mean": _mean("prompt_tokens", guidance),
        "completion_tokens_per_query_mean": _mean("completion_tokens", guidance),
        "total_tokens_per_query_mean": _mean("total_tokens", guidance),
        "latency_seconds_per_query_mean": _mean("latency_seconds", guidance),
    }
    payload = {"artifact_format": "lgbo-llm-telemetry-v1",
               "summary": summary, "per_call": calls}
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except OSError as exc:  # never let telemetry I/O break a completed run
        log("LGBO telemetry write failed (%s)" % exc)


class _GP(common._GP):
    """Matérn-5/2 GP with per-iteration marginal-likelihood fitting."""

    def fit(self, X, y, optimize=True):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[finite], y[finite]
        if X.shape[0] < 1:
            return super().fit(X, y)

        self.X_train_ = X.copy()
        self.y_mean_ = float(np.mean(y))
        centered = y - self.y_mean_
        scale = float(np.std(centered))
        self.y_scale_ = scale if scale > 1e-12 else 1.0
        target = centered / self.y_scale_

        if optimize and X.shape[0] >= 2:
            from scipy.optimize import minimize

            initial = np.log([self.length_scale, self.signal_variance, self.noise_variance])
            bounds = [(np.log(0.02), np.log(2.0)),
                      (np.log(1e-3), np.log(100.0)),
                      (np.log(1e-8), np.log(0.25))]

            def nll(theta):
                self.length_scale, self.signal_variance, self.noise_variance = np.exp(theta)
                K = self._kernel(X, X) + (self.noise_variance + 1e-9) * np.eye(X.shape[0])
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, target))
                except (np.linalg.LinAlgError, ValueError):
                    return 1e30
                value = (0.5 * target @ alpha + np.log(np.diag(L)).sum()
                         + 0.5 * X.shape[0] * np.log(2.0 * np.pi))
                return float(value) if np.isfinite(value) else 1e30

            result = minimize(nll, initial, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 100})
            theta = result.x if np.all(np.isfinite(result.x)) else initial
            self.length_scale, self.signal_variance, self.noise_variance = np.exp(theta)

        # The parent recomputes normalization, then factors with the selected parameters.
        return super().fit(X, y)

    def posterior_cross_cov(self, X, G):
        """Current posterior cross-covariance Sigma(X,G), in standardized units."""

        X = np.atleast_2d(np.asarray(X, dtype=float))
        G = np.atleast_2d(np.asarray(G, dtype=float))
        Kxg = self._kernel(X, G)
        if getattr(self, "_degenerate_", False) or self.X_train_.shape[0] == 0:
            return Kxg
        Kxt = self._kernel(X, self.X_train_)
        Ktg = self._kernel(self.X_train_, G)
        vx = np.linalg.solve(self.L_, Kxt.T)
        vg = np.linalg.solve(self.L_, Ktg)
        return Kxg - vx.T @ vg

    def posterior_cov(self, X):
        return self.posterior_cross_cov(X, X)


def _regional_shift_z(gp, X, G, a, confidence, eps=1e-12):
    """Return (shift, q, lambda) for the exact posterior exponential tilt."""

    a = np.asarray(a, dtype=float).reshape(-1)
    sigma_gg = gp.posterior_cov(G)
    q = float(a @ sigma_gg @ a)
    if not np.isfinite(q) or q <= eps or confidence <= 0.0:
        return np.zeros(np.atleast_2d(X).shape[0]), q, 0.0
    lam = float(confidence) / np.sqrt(q)
    shift = lam * (gp.posterior_cross_cov(X, G) @ a)
    if not np.all(np.isfinite(shift)):
        return np.zeros(np.atleast_2d(X).shape[0]), q, 0.0
    return shift, q, lam


def _lgbo_step(gp, encoded, evaluated_mask, guidance, n_grid, seed, decay,
               best_observed):
    shift_z = np.zeros(encoded.shape[0])
    q = lam = 0.0
    if guidance is not None:
        kind, lb, ub, confidence = guidance
        G, a = common._grid_and_weights(kind, lb, ub, encoded.shape[1], n_grid, seed, decay)
        shift_z, q, lam = _regional_shift_z(gp, encoded, G, a, confidence)

    mean, std = gp.predict(encoded)
    mean = mean + shift_z * gp.y_scale_
    std = np.maximum(std, 0.0)  # the exponential tilt does not change covariance
    improvement = mean - float(best_observed)
    ei = np.maximum(improvement, 0.0)
    positive = std > 1e-12
    if np.any(positive):
        z = improvement[positive] / std[positive]
        cdf = 0.5 * (1.0 + common._erf_vec(z / np.sqrt(2.0)))
        pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
        ei[positive] = improvement[positive] * cdf + std[positive] * pdf
    # q=1 log-EI is monotone in EI; logging is only a numerically stable transform.
    log_ei = np.log(np.maximum(ei, np.finfo(float).tiny))
    log_ei = np.nan_to_num(log_ei, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    log_ei[evaluated_mask] = -np.inf
    return int(np.argmax(log_ei)), q, lam


def _initial_prompt(dim, config, n_points):
    direction = "MINIMIZE" if config.get("lower_is_better", True) else "MAXIMIZE"
    context = str(config.get("description", "")).strip()
    return (
        "Choose exactly %d informative initial evaluation locations for Bayesian optimization "
        "of %s (%s); we want to %s it. Coordinates are normalized to [0,1] in %d dimensions.\n"
        % (n_points, config.get("model", "a black-box objective"),
           config.get("metric", "objective value"), direction, dim)
        + (context + "\n" if context else "")
        + 'Return only JSON: {"points": [[x0,...], ...]}. Use exactly %d distinct points.'
        % n_points
    )


def _parse_initial_points(raw, dim, n_points):
    try:
        start = raw.index("{")
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
        points = np.asarray(obj["points"], dtype=float)
    except (ValueError, KeyError, TypeError):
        return None
    if points.shape != (n_points, dim) or not np.all(np.isfinite(points)):
        return None
    return np.clip(points, 0.0, 1.0)


def _llm_initial_indices(chat, encoded, config, n_points, seed):
    try:
        reply = chat([{"role": "user", "content": _initial_prompt(
            encoded.shape[1], config, n_points)}], temperature=0.0,
            max_tokens=int(config.get("max_tokens", 512)), kind="initial", query_after=0)
        points = _parse_initial_points(reply, encoded.shape[1], n_points)
    except Exception as exc:
        log("LGBO initial LLM call failed (%s); using seeded fallback" % exc)
        points = None
    if points is None:
        return seeded_initial_indices(encoded.shape[0], n_points, seed)
    chosen = []
    unavailable = np.zeros(encoded.shape[0], dtype=bool)
    for point in points:
        distances = np.square(encoded - point).sum(axis=1)
        distances[unavailable] = np.inf
        index = int(np.argmin(distances))
        chosen.append(index)
        unavailable[index] = True
    return chosen


def main():
    spec = read_spec()
    encoded = normalize_rows(spec["encoded"])
    budget = int(spec["budget"])
    seed = int(spec.get("seed") or 0)
    maximize = bool(spec.get("maximize", True))
    config = dict(spec.get("config") or {})
    evaluate = make_evaluator()
    n_total, dim = encoded.shape
    n_init = min(max(1, int(config.get("paper_n_initial", 2))), budget, n_total)
    config.setdefault("lower_is_better", not maximize)

    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
    model = str(config.get("llm_model") or os.environ.get("LLM_MODEL") or "local-model")
    chat = _make_instrumented_chat(
        base_url=base_url, api_key=os.environ.get("LLM_API_KEY", "none"),
        model=model, timeout=float(config.get("request_timeout", 300.0)),
        seed=seed, sink=_TELEMETRY)
    initial_indices = _llm_initial_indices(chat, encoded, config, n_init, seed)
    log("LGBO endpoint=%s model=%s paper_n_init=%d budget=%d n_grid=%d dim=%d"
        % (base_url, model, n_init, budget, int(config.get("n_grid", 64)), dim))

    evaluated = np.zeros(n_total, dtype=bool)
    X_obs, y_obs, history = [], [], []

    def observe(index):
        raw = float(evaluate(index))
        evaluated[index] = True
        X_obs.append(encoded[index].tolist())
        y_obs.append(raw if maximize else -raw)
        history.append(("[" + ", ".join("%.4f" % v for v in encoded[index]) + "]", raw))

    for index in initial_indices:
        if len(y_obs) < budget:
            observe(index)

    while len(y_obs) < budget and not np.all(evaluated):
        index = None
        try:
            gp = _GP(length_scale=float(config.get("length_scale", 0.2)),
                     noise_std=float(config.get("noise_std", 1e-3))).fit(
                         np.asarray(X_obs), np.asarray(y_obs))
            lines = ["  %s -> %.6g" % row for row in history[-30:]]
            guidance = None
            try:
                reply = chat([{"role": "user", "content": common._guidance_prompt(
                    dim, config, lines)}], temperature=0.0,
                    max_tokens=int(config.get("max_tokens", 512)),
                    kind="guidance", query_after=len(y_obs))
                guidance = common._parse_guidance(reply, dim)
                if guidance is None:
                    log("LGBO guidance parse failed; using unshifted log-EI")
            except Exception as exc:
                log("LGBO guidance call failed (%s); using unshifted log-EI" % exc)
            index, q, lam = _lgbo_step(
                gp, encoded, evaluated, guidance, int(config.get("n_grid", 64)),
                seed + 1000 + len(y_obs), float(config.get("decay", 4.0)), max(y_obs))
            if guidance is not None:
                log("LGBO posterior lift q=%.6g lambda=%.6g confidence=%.4g"
                    % (q, lam, guidance[3]))
        except Exception as exc:
            log("LGBO step failed (%s); using unseen fallback" % exc)
        if index is None or index < 0 or evaluated[index]:
            remaining = np.flatnonzero(~evaluated)
            if remaining.size == 0:
                break
            index = int(remaining[0])
        observe(index)
    _write_telemetry(config, budget, dim, seed)
    finish()


if __name__ == "__main__":
    main()
