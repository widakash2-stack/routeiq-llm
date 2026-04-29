import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'crypto_payment_intelligence', 'src'))

from openai import OpenAI
from routing_engine import route_transaction_with_trace

# ── CONFIG ──────────────────────────────────────────
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── PROMPT TEMPLATE ─────────────────────────────────
SYSTEM_PROMPT = """You are a payments operations analyst explaining
routing decisions to a non-technical ops manager.

Rules:
- Maximum 3 sentences
- No technical jargon (no alpha/beta, no bandit, no UCB, no EMA)
- Always state which PSP was selected and the main reason why
- Always end with a clear action: "No action needed" or "Escalate to engineering"
- Use plain numbers: percentages and dollar amounts only"""

def explain_routing_decision(transaction: dict) -> str:
    """
    Takes a transaction dict, runs it through RouteIQ engine,
    and returns a plain English explanation for ops managers.
    """
    # Step 1 — Get routing decision + trace from your engine
    selected_psp, trace = route_transaction_with_trace(transaction)

    # Step 2 — Build context for the LLM
    psp_ranking = trace.get("psp_ranking", [])
    top_psps = psp_ranking[:3]  # top 3 PSPs and their scores

    ranking_text = "\n".join([
        f"  - {p['psp']}: success score {round(p['local_sample']*100, 1)}%, "
        f"cost score {round(p['cost_score']*100, 1)}%, "
        f"latency score {round(p['latency_score']*100, 1)}%"
        for p in top_psps
    ])

    user_prompt = f"""A payment routing decision was just made. Explain it clearly.

Transaction details:
- Country: {trace['context']['country']}
- Payment method: {trace['context']['payment_method']}
- Selected PSP: {selected_psp}
- Reason engine selected it: {trace['why_winner']}
- Selection reason: {trace['reason']}

Top PSPs considered:
{ranking_text}

Write a 2-3 sentence explanation for an ops manager."""

    # Step 3 — Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=150,
        temperature=0.3  # low temperature = consistent, factual outputs
    )

    return response.choices[0].message.content.strip()


# ── TEST IT ──────────────────────────────────────────
if __name__ == "__main__":

    # Test Case 1 — Normal transaction
    test_transaction = {
        "country":        "NGN",
        "payment_method": "BANK_TRANSFER",
        "amount":         500,
        "time_bucket":    "afternoon"
    }

    print("=" * 60)
    print("ROUTEIQ LLM EXPLANATION LAYER — TEST RUN")
    print("=" * 60)
    print(f"Transaction: {test_transaction}")
    print()

    explanation = explain_routing_decision(test_transaction)

    print("EXPLANATION FOR OPS MANAGER:")
    print("-" * 40)
    print(explanation)
    print("-" * 40)
