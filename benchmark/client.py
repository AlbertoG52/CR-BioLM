"""One chat-completions call, with everything the answer row needs to be auditable.

Returns the text plus the metadata that makes a run reproducible rather than merely
repeatable: which upstream served it, whether the pin held, tokens, cost and latency.

Determinism: temperature 0 and a fixed seed are sent when the model accepts them. Both
chosen models do (that is partly why they were chosen -- the Claude 5 flagships accept
neither). A model that silently ignores them still gets its variability measured by the
replicate subset, so this is best-effort rather than a guarantee.
"""
from __future__ import annotations

import time

import requests

from llm import providers as P

SEED = 20260913          # same seed as the species sample, for no reason but tidiness
TIMEOUT_S = 180
RETRIES = 3
BACKOFF_S = 4

# Generous, because the paid generator is a REASONING model: on a short grounded
# question it spent 301 of 335 completion tokens thinking, and at max_tokens=200 the
# reasoning consumed the whole budget and the content came back empty. An empty answer
# that scores as an abstention would be a silent data-quality failure, so the budget is
# wide and `_empty_answer` below turns a blank completion into an error instead.
MAX_TOKENS = 2400


class CallError(RuntimeError):
    pass


def call(spec: str, messages: list[dict], *, pin: tuple[str, ...] = (),
         temperature: float = 0.0, max_tokens: int = MAX_TOKENS,
         seed: int | None = SEED) -> dict:
    """Issue one call. Raises CallError after RETRIES failures."""
    prov = P.resolve(spec, pin=pin)
    ok, why = P.is_configured(spec)
    if not ok:
        raise CallError(f"{spec}: {why}")

    body = {"model": prov.model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    if seed is not None:
        body["seed"] = seed
    body.update(prov.routing())
    if prov.name == "openrouter":
        # Ask OpenRouter to return its own cost accounting rather than inferring it
        # from a price table that may have moved since the run.
        body["usage"] = {"include": True}

    last = None
    for attempt in range(1, RETRIES + 1):
        t0 = time.time()
        try:
            r = requests.post(prov.url, headers=prov.headers(), json=body, timeout=TIMEOUT_S)
            latency_ms = int((time.time() - t0) * 1000)
            if r.status_code != 200:
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                # 520-524 are the gateway's "upstream misbehaved" family; like the
                # others they clear on a retry.
                if r.status_code in (408, 429, 500, 502, 503, 504, 520, 522, 524):
                    time.sleep(BACKOFF_S * attempt)
                    continue
                raise CallError(f"{spec}: {last}")
            j = r.json()
            choices = j.get("choices") or []
            if not choices:
                last = f"no choices: {str(j)[:300]}"
                time.sleep(BACKOFF_S * attempt)
                continue
            text = (choices[0].get("message") or {}).get("content") or ""
            usage = j.get("usage") or {}
            served = P.served_by(j)
            finish = choices[0].get("finish_reason")
            if not text.strip():
                # Never record a blank as an answer: it would score as an abstention and
                # quietly become a result. Retry, then fail loudly.
                last = (f"empty content (finish_reason={finish}, "
                        f"reasoning_tokens="
                        f"{(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')})")
                if attempt < RETRIES:
                    body["max_tokens"] = int(body.get("max_tokens", MAX_TOKENS) * 1.5)
                    time.sleep(BACKOFF_S)
                    continue
                raise CallError(f"{spec}: {last}")
            return {
                "answer_text": text.strip(),
                "model_id": spec,
                "model_version": j.get("model") or prov.model,
                "provider_pinned": ",".join(pin) if pin else None,
                "served_by": served,
                "pin_held": (None if not pin else
                             (served is not None and served.lower() in
                              {p.lower() for p in pin})),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {})
                                    .get("reasoning_tokens"),
                "finish_reason": finish,
                "cost_usd": usage.get("cost"),
                "latency_ms": latency_ms,
                "attempts": attempt,
            }
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(BACKOFF_S * attempt)
    raise CallError(f"{spec}: failed after {RETRIES} attempts -- {last}")
