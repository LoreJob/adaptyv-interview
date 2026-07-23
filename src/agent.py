"""Round Advisor agent — natural-language orchestration over the model + mock API.

A thin tool-calling agent that turns requests like *"I have budget for 20 tests,
give me the best selection and explain the trade-off"* into calls against four tools:

* ``predict_affinity(sequences)`` — binder probability + KD estimate per sequence.
* ``rank_candidates(sequences, budget)`` — UCB-rank and return the top ``budget``.
* ``submit_batch(sequences)`` — submit to the mock Adaptyv API (simulated lab).
* ``summarize_round(batch_id)`` — summarize a submitted batch's results.

Kept deliberately small (4 tools): a simple agent that works beats an ambitious
one that breaks in a demo. Model + KD estimates are illustrative, and the API
schema is hypothetical — the agent's system prompt says so.

**Backend: OpenRouter** via the OpenAI-compatible SDK (OpenRouter does not
implement Anthropic's native Messages API / Tool Runner, so we run a manual
tool-calling loop). Requires ``OPENROUTER_API_KEY`` (via env or a `.env` file).
Model is set by ``OPENROUTER_MODEL`` (default ``anthropic/claude-sonnet-4.5``) —
change it to any Claude slug your OpenRouter account can access.
"""

from __future__ import annotations

import json
import os

if __package__ in (None, ""):  # allow `python src/agent.py` and IDE Run
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from src import mock_api

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
MAX_TOOL_ITERATIONS = 8  # safety cap on the agentic loop

SYSTEM_PROMPT = """\
You are Round Advisor, an assistant for protein-binder design campaigns at a \
cloud lab (modeled on Adaptyv Bio's EGFR competitions). You help scientists spend \
a fixed experimental budget well: predicting which designed sequences are likely \
binders, ranking candidates under a budget, submitting batches to the lab API, and \
summarizing results.

Be honest about limits:
- The predictive model is trained on a small public dataset (~600 designs, 63 \
binders). Predictions are illustrative, not production-grade. Don't overstate \
accuracy in front of people who generate this data for a living.
- KD estimates and the lab API are simulated/hypothetical, not real measurements \
or Adaptyv's real API schema.

When a user gives a budget, use rank_candidates to pick the top designs, then \
explain the trade-off (informed selection vs testing everything / random). Keep \
answers concise and quantitative.
"""


# --- tool implementations -------------------------------------------------

def _predict_affinity(sequences: list[str]) -> str:
    return json.dumps(mock_api.score_sequences(sequences), indent=2)


def _rank_candidates(sequences: list[str], budget: int) -> str:
    scored = mock_api.score_sequences(sequences)
    scored.sort(key=lambda d: d["acquisition_score"], reverse=True)
    return json.dumps(
        {"budget": budget, "selected": scored[: max(0, budget)], "n_total": len(scored)},
        indent=2,
    )


def _submit_batch(sequences: list[str], round_name: str = "") -> str:
    resp = mock_api.submit_batch(sequences, round_name=round_name or None)
    return resp.model_dump_json(indent=2)


def _summarize_round(batch_id: str) -> str:
    try:
        resp = mock_api.get_results(batch_id)
    except KeyError:
        return json.dumps({"error": f"Unknown batch_id: {batch_id}"})
    probs = [r.binder_probability for r in resp.results]
    likely = [r for r in resp.results if r.binder_probability >= 0.5]
    return json.dumps({
        "batch_id": batch_id,
        "n_designs": len(resp.results),
        "n_likely_binders": len(likely),
        "mean_binder_probability": round(sum(probs) / len(probs), 4) if probs else None,
        "top": [
            {"binder_probability": r.binder_probability,
             "predicted_log10_kd": r.predicted_log10_kd,
             "sequence": r.sequence[:40] + ("..." if len(r.sequence) > 40 else "")}
            for r in sorted(resp.results, key=lambda r: r.binder_probability, reverse=True)[:5]
        ],
    }, indent=2)


_DISPATCH = {
    "predict_affinity": _predict_affinity,
    "rank_candidates": _rank_candidates,
    "submit_batch": _submit_batch,
    "summarize_round": _summarize_round,
}

# OpenAI-format tool schemas (OpenRouter passes these through to the model).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_affinity",
            "description": "Predict binder probability, its uncertainty, and estimated log10(KD in molar) for protein sequences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequences": {"type": "array", "items": {"type": "string"},
                                  "description": "Protein sequences in one-letter amino-acid code."}
                },
                "required": ["sequences"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_candidates",
            "description": "Rank candidate sequences by a UCB acquisition score (predicted binder probability + uncertainty) and return the top `budget`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequences": {"type": "array", "items": {"type": "string"},
                                  "description": "Candidate protein sequences in one-letter code."},
                    "budget": {"type": "integer", "description": "How many designs the user can afford to test."},
                },
                "required": ["sequences", "budget"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_batch",
            "description": "Submit a batch of designs to the (mock) Adaptyv lab API. Results are simulated, not real measurements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequences": {"type": "array", "items": {"type": "string"},
                                  "description": "Protein sequences to submit for testing."},
                    "round_name": {"type": "string", "description": "Optional label for this round."},
                },
                "required": ["sequences"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_round",
            "description": "Summarize the results of a previously submitted batch, by batch id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {"type": "string", "description": "Batch id returned by submit_batch."}
                },
                "required": ["batch_id"],
            },
        },
    },
]


def _make_client() -> OpenAI:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (put it in .env).")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


def run_agent(user_message: str, client: OpenAI | None = None) -> str:
    """Run the agent on one request; drive the tool loop; return the final text."""
    client = client or _make_client()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or ""

        # Execute each requested tool, append results, loop.
        for call in msg.tool_calls:
            fn = _DISPATCH.get(call.function.name)
            try:
                args = json.loads(call.function.arguments or "{}")
                result = fn(**args) if fn else json.dumps({"error": f"unknown tool {call.function.name}"})
            except Exception as exc:  # surface tool errors back to the model
                result = json.dumps({"error": str(exc)})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    return "Stopped: exceeded max tool iterations."


if __name__ == "__main__":
    import sys
    prompt = " ".join(sys.argv[1:]) or (
        "I have budget to test 3 of these 5 designs. Pick the best and explain the "
        "trade-off vs testing all 5.\n"
        "SEQ1: QVQLQESGGGLVQPGGSLRLSCAASGRTFSSYAMGWFRQAPGKQREFVAAIRWSGGYTYYTDSVKGRFTISRDNAKTTVYLQMNSLKPEDTAVYYCAATYLSSDYSRYALPQRPLDYDYWGQGTQVTVSS\n"
        "SEQ2: MKKLLPTAAAGLLLLAAQPAMA\n"
        "SEQ3: EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDLGRRGYFDYWGQGTLVTVSS\n"
        "SEQ4: GSHMKEIAALKEKIAALKEKIAALKE\n"
        "SEQ5: DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK"
    )
    print(run_agent(prompt))
