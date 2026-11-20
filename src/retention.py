"""AI Retention Action Generator: drafts a tailored offer from a customer's top SHAP drivers.

Two engines, one output shape:

* **Template engine** (default, deterministic, offline): maps each driver to a catalogue
  action and prices it against the per-customer offer budget.
* **LLM engine** (optional - Gemini / OpenAI / Anthropic): the model may only *choose from a
  pre-priced menu* of catalogue actions and *word* the message. Prices are computed here,
  never by the model. Any problem - no key, timeout, API error, bad JSON, off-menu action,
  over-budget, banned wording - silently falls back to the template engine, and the result
  records why.

The customer-facing message never mentions churn, scores or models, and never refers to
demographics (those are already filtered out of the drivers upstream).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)

MAX_ACTIONS = 3
MONEY_SHARE_OF_BUDGET = 0.6  # at most 60% of the budget on a single bill concession
OFFER_VALID_DAYS = 14
LLM_TIMEOUT_SECONDS = 25

# Assumed internal cost of each add-on per month, used only to price free trials.
ADDON_MONTHLY_COST = {
    "tech_support": 10.0,
    "online_security": 8.0,
    "online_backup": 6.0,
    "device_protection": 7.0,
}
ADDON_LABEL = {
    "tech_support": "Premium Tech Support",
    "online_security": "Online Security",
    "online_backup": "Online Backup",
    "device_protection": "Device Protection",
}
ADDON_TRIAL_MONTHS = 3

# Wording the customer-facing message must never contain.
BANNED_PATTERNS = [
    r"\bchurn", r"\bat[- ]risk\b", r"\bshap\b", r"\bmodel\b", r"\bscore", r"\bprobabilit",
    r"\bpredict", r"\blikely to (leave|cancel)", r"\bgender\b", r"\bage\b", r"\bsenior",
    r"\bmarried\b", r"\bdependents?\b", r"\bpartner\b",
]  # fmt: skip
_BANNED = re.compile("|".join(BANNED_PATTERNS), re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Catalogue of actions
# --------------------------------------------------------------------------- #
Offer = dict[str, Any]  # id, kind, cost, short, customer_text, rep_text
Builder = Callable[[Mapping[str, Any], float], "Offer | None"]


def _offer(oid: str, kind: str, cost: float, short: str, customer: str, rep: str) -> Offer:
    return {
        "id": oid,
        "kind": kind,
        "cost": round(float(cost), 2),
        "short": short,
        "customer_text": customer,
        "rep_text": rep,
    }


def _contract_discount(row: Mapping[str, Any], cap: float) -> Offer | None:
    contract = row.get("contract")
    monthly = float(row.get("monthly_charges", 0) or 0)
    if contract == "Two Year" or monthly <= 0:
        return None
    months, max_pct = 12, 0.15
    pct = math.floor(min(max_pct, cap / (monthly * months)) * 100 + 1e-9) / 100
    if pct < 0.03:
        return None
    term = "1-year" if contract == "Month-to-Month" else "2-year"
    cost = pct * monthly * months
    return _offer(
        "contract_discount",
        "money",
        cost,
        f"{term} plan at {pct:.0%} off",
        f"{pct:.0%} off your monthly bill for the next {months} months when you move to a "
        f"{term} plan",
        f"Move customer from {contract} to a {term} contract with {pct:.0%} off for "
        f"{months} months (est. cost ${cost:,.0f}).",
    )


def _loyalty_credit(row: Mapping[str, Any], cap: float) -> Offer | None:
    monthly = float(row.get("monthly_charges", 0) or 0)
    credit = math.floor(min(0.5 * monthly, cap, 50.0) / 5) * 5
    if credit < 5:
        return None
    return _offer(
        "loyalty_credit",
        "money",
        credit,
        f"${credit:.0f} loyalty credit",
        f"a one-time ${credit:.0f} loyalty credit on your next bill",
        f"Apply a one-time ${credit:.0f} bill credit (price sensitivity signal).",
    )


def _fixed(
    oid: str, cost: float, short: str, customer: str, rep: str, requires_internet: bool = False
) -> Builder:
    def build(row: Mapping[str, Any], cap: float) -> Offer | None:
        if requires_internet and not row.get("internet_service", 1):
            return None
        if cost > cap + 1e-9:
            return None
        return _offer(oid, "service", cost, short, customer, rep)

    return build


def _addon(feature: str) -> Builder:
    label, months = ADDON_LABEL[feature], ADDON_TRIAL_MONTHS
    cost = ADDON_MONTHLY_COST[feature] * months

    def build(row: Mapping[str, Any], cap: float) -> Offer | None:
        if not row.get("internet_service", 1) or row.get(feature, 1) != 0 or cost > cap + 1e-9:
            return None
        return _offer(
            f"addon_{feature}",
            "service",
            cost,
            f"{months} free months of {label}",
            f"{months} months of {label} on us",
            f"Free {months}-month {label} trial (customer has none; est. cost ${cost:,.0f}).",
        )

    return build


def _addon_any(row: Mapping[str, Any], cap: float) -> Offer | None:
    for feature in ("online_security", "tech_support", "online_backup", "device_protection"):
        offer = _addon(feature)(row, cap)
        if offer:
            return offer
    return None


def _fiber_check(row: Mapping[str, Any], cap: float) -> Offer | None:
    if row.get("internet_type") != "Fiber Optic" or 10.0 > cap + 1e-9:
        return None
    return _offer(
        "fiber_check",
        "service",
        10.0,
        "free speed & connection check",
        "a free speed and connection check, with a priority repair slot if anything is off",
        "Book a proactive fiber speed/connection health check with priority repair.",
    )


def _autopay_credit(row: Mapping[str, Any], cap: float) -> Offer | None:
    if row.get("payment_method") != "Mailed Check" or 5.0 > cap + 1e-9:
        return None
    return _offer(
        "autopay_credit",
        "money",
        5.0,
        "$5 autopay credit",
        "a $5 credit when you switch to automatic payments",
        "Offer a $5 credit to move from mailed check to automatic payment.",
    )


_ONBOARDING = _fixed(
    "onboarding_call",
    5.0,
    "personal onboarding call",
    "a personal call from your account team to make sure everything is set up the way you want",
    "Schedule a personal onboarding / check-in call (est. labour $5).",
)
_PLAN_REVIEW = _fixed(
    "plan_review",
    0.0,
    "free plan review",
    "a complimentary plan review to make sure you are only paying for what you use",
    "Offer a free plan right-sizing review.",
)
_BILLING_WALKTHROUGH = _fixed(
    "billing_walkthrough",
    0.0,
    "bill walkthrough",
    "a quick walkthrough of your bill so every charge is clear",
    "Offer a plain-language bill walkthrough.",
)
_REFERRALS = _fixed(
    "referral_rewards",
    0.0,
    "referral rewards",
    "a $10 credit for every friend you refer",
    "Enrol in referral rewards ($10 per referral; cost only incurred on a referral).",
)
_ACCOUNT_CALL = _fixed(
    "account_manager_call",
    0.0,
    "account-manager check-in",
    "a check-in call from your account manager to hear how we can serve you better",
    "Personal check-in call from the account manager.",
)

ACTIONS: dict[str, Builder] = {
    "contract_discount": _contract_discount,
    "loyalty_credit": _loyalty_credit,
    "onboarding_call": _ONBOARDING,
    "fiber_check": _fiber_check,
    "addon_tech_support": _addon("tech_support"),
    "addon_online_security": _addon("online_security"),
    "addon_online_backup": _addon("online_backup"),
    "addon_device_protection": _addon("device_protection"),
    "addon_any": _addon_any,
    "autopay_credit": _autopay_credit,
    "billing_walkthrough": _BILLING_WALKTHROUGH,
    "referral_rewards": _REFERRALS,
    "plan_review": _PLAN_REVIEW,
    "account_manager_call": _ACCOUNT_CALL,
}

_PRICE_FEATURES = [
    "monthly_charges",
    "avg_monthly_spend",
    "charge_vs_tenure_gap",
    "total_charges",
    "total_long_distance_charges",
    "avg_monthly_long_distance_charges",
    "total_extra_data_charges",
]
# driver feature -> ordered list of candidate action ids (first feasible wins)
DRIVER_ACTIONS: dict[str, list[str]] = {
    "contract": ["contract_discount", "plan_review"],
    "tenure": ["onboarding_call", "account_manager_call"],
    "is_new_fiber_m2m": ["fiber_check", "onboarding_call", "contract_discount"],
    "internet_type": ["fiber_check", "plan_review"],
    "tech_support": ["addon_tech_support", "plan_review"],
    "online_security": ["addon_online_security", "plan_review"],
    "online_backup": ["addon_online_backup", "plan_review"],
    "device_protection": ["addon_device_protection", "plan_review"],
    "service_count": ["addon_any", "plan_review"],
    "payment_method": ["autopay_credit", "billing_walkthrough"],
    "paperless_billing": ["billing_walkthrough"],
    "number_of_referrals": ["referral_rewards"],
    "referred_a_friend": ["referral_rewards"],
    **{f: ["loyalty_credit", "plan_review"] for f in _PRICE_FEATURES},
    **{
        f: ["plan_review"]
        for f in (
            "streaming_tv",
            "streaming_movies",
            "streaming_music",
            "unlimited_data",
            "avg_monthly_gb_download",
            "phone_service",
            "multiple_lines",
            "internet_service",
        )
    },
}


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #
@dataclass
class RetentionOffer:
    headline: str
    customer_message: str
    rep_notes: str
    actions: list[Offer]
    total_cost: float
    budget: float
    generated_by: str  # "Template" | "LLM"
    provider: str | None = None
    model: str | None = None
    fallback_reason: str | None = None
    drivers: list[dict[str, Any]] = field(default_factory=list)

    @property
    def within_budget(self) -> bool:
        return self.total_cost <= self.budget + 1e-6

    @property
    def badge(self) -> str:
        if self.generated_by == "LLM":
            return f"Generated by: LLM ({self.provider})"
        return "Generated by: Template"


# --------------------------------------------------------------------------- #
# Template engine
# --------------------------------------------------------------------------- #
def _builders_for(feature: str) -> list[str]:
    return DRIVER_ACTIONS.get(feature, [])


def candidate_menu(
    row: Mapping[str, Any], drivers: Sequence[Mapping[str, Any]], budget: float
) -> list[Offer]:
    """Every distinct catalogue action relevant to these drivers, each priced on its own.

    This is the menu the LLM chooses from. Pricing is done here, in code.
    """
    money_cap = MONEY_SHARE_OF_BUDGET * budget
    money_ids = {"contract_discount", "loyalty_credit", "autopay_credit"}
    menu: list[Offer] = []
    seen: set[str] = set()
    for driver in drivers:
        for action_id in _builders_for(driver["feature"]):
            cap = money_cap if action_id in money_ids else budget
            offer = ACTIONS[action_id](row, cap)
            if offer and offer["id"] not in seen and offer["cost"] <= budget + 1e-9:
                offer["driver_feature"] = driver["feature"]
                menu.append(offer)
                seen.add(offer["id"])
    if "account_manager_call" not in seen:
        menu.append(ACTIONS["account_manager_call"](row, budget))  # type: ignore[arg-type]
    return menu


def select_actions(
    row: Mapping[str, Any], drivers: Sequence[Mapping[str, Any]], budget: float
) -> list[Offer]:
    """Deterministic choice: walk drivers strongest-first, take the first feasible action each."""
    remaining = float(budget)
    chosen: list[Offer] = []
    ids: set[str] = set()
    money_used = False
    for driver in drivers:
        if len(chosen) >= MAX_ACTIONS:
            break
        for action_id in _builders_for(driver["feature"]):
            builder = ACTIONS[action_id]
            probe = builder(row, remaining)
            if probe is None:
                continue
            if probe["kind"] == "money":
                if money_used:
                    continue
                probe = builder(row, min(remaining, MONEY_SHARE_OF_BUDGET * budget))
                if probe is None:
                    continue
            if probe["id"] in ids or probe["cost"] > remaining + 1e-9:
                continue
            probe["driver_feature"] = driver["feature"]
            chosen.append(probe)
            ids.add(probe["id"])
            remaining -= probe["cost"]
            money_used = money_used or probe["kind"] == "money"
            break
    if not chosen:
        fallback = ACTIONS["account_manager_call"](row, remaining)
        assert fallback is not None
        fallback["driver_feature"] = drivers[0]["feature"] if drivers else None
        chosen.append(fallback)
    return chosen


def _join_phrases(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def compose_customer_message(actions: Sequence[Offer]) -> str:
    perks = _join_phrases([a["customer_text"] for a in actions])
    return (
        "Hello, and thank you for being a customer. We would like to make your service work "
        f"even better for you, and to say thanks we are offering {perks}. "
        f"This offer is open for the next {OFFER_VALID_DAYS} days: just reply to this "
        "message or call your account team and we will set it up."
    )


def compose_rep_notes(
    actions: Sequence[Offer],
    drivers: Sequence[Mapping[str, Any]],
    total_cost: float,
    budget: float,
    context: Mapping[str, Any],
) -> str:
    text_by_feature = {d["feature"]: d["text"] for d in drivers}
    lines = []
    for a in actions:
        why = text_by_feature.get(a.get("driver_feature") or "", "")
        prefix = f"{why} -> " if why else ""
        lines.append(f"- {prefix}{a['rep_text']}")
    head = []
    if "p_churn" in context:
        head.append(f"Risk {context['p_churn']:.0%}")
    if "ltv" in context:
        head.append(f"LTV ${context['ltv']:,.0f}")
    head.append(f"est. offer cost ${total_cost:,.0f} of ${budget:,.0f} budget")
    return " | ".join(head) + "\n" + "\n".join(lines)


def template_offer(
    row: Mapping[str, Any],
    drivers: Sequence[Mapping[str, Any]],
    budget: float,
    context: Mapping[str, Any] | None = None,
) -> RetentionOffer:
    actions = select_actions(row, drivers, budget)
    total = sum(a["cost"] for a in actions)
    return RetentionOffer(
        headline=" + ".join(a["short"] for a in actions),
        customer_message=compose_customer_message(actions),
        rep_notes=compose_rep_notes(actions, drivers, total, budget, context or {}),
        actions=actions,
        total_cost=round(total, 2),
        budget=float(budget),
        generated_by="Template",
        drivers=[dict(d) for d in drivers],
    )


# --------------------------------------------------------------------------- #
# LLM providers (all imports are lazy; every provider is optional)
# --------------------------------------------------------------------------- #
class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, user: str) -> str: ...


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self.api_key = api_key
        self.model = model or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

    def generate(self, system: str, user: str) -> str:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        resp = client.models.generate_content(
            model=self.model,
            contents=user,
            config={"system_instruction": system, "response_mime_type": "application/json"},
        )
        return resp.text or ""


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self.api_key = api_key
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    def generate(self, system: str, user: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, timeout=LLM_TIMEOUT_SECONDS)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str | None = None) -> None:
        self.api_key = api_key
        self.model = model or os.getenv("ANTHROPIC_MODEL") or "claude-haiku-5-5"

    def generate(self, system: str, user: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key, timeout=LLM_TIMEOUT_SECONDS)
        resp = client.messages.create(
            model=self.model,
            max_tokens=900,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


_PROVIDERS: dict[str, tuple[type, str]] = {
    "gemini": (GeminiProvider, "GEMINI_API_KEY"),
    "openai": (OpenAIProvider, "OPENAI_API_KEY"),
    "anthropic": (AnthropicProvider, "ANTHROPIC_API_KEY"),
}


def get_provider(name: str | None = None, env: Mapping[str, str] | None = None):
    """Provider instance for ``name`` (default: $LLM_PROVIDER), or None -> template engine.

    None is returned for "template", an unknown name, or a missing API key.
    """
    env = os.environ if env is None else env
    name = (name or env.get("LLM_PROVIDER") or "template").strip().lower()
    if name not in _PROVIDERS:
        return None
    cls, key_var = _PROVIDERS[name]
    key = (env.get(key_var) or "").strip()
    return cls(key) if key else None


# --------------------------------------------------------------------------- #
# LLM engine with validation + fallback
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You write retention offers for a telecom Customer Success team. You will receive a "
    "customer summary, the reasons the account needs attention, a budget, and a MENU of "
    "pre-priced actions. Choose 1 to 3 actions from the menu by id and write the message. "
    "Rules: use only menu ids; never invent actions or prices; the chosen costs must sum to "
    "no more than the budget; the customer_message is shown to the customer, so never mention "
    "churn, risk, scores, models, predictions, or personal attributes such as age, gender, "
    "family status. Keep customer_message under 600 characters, warm and specific. "
    'Reply with JSON only: {"action_ids": [...], "headline": "...", '
    '"customer_message": "...", "rep_notes": "..."}'
)


def build_llm_prompt(
    row: Mapping[str, Any],
    drivers: Sequence[Mapping[str, Any]],
    menu: Sequence[Offer],
    budget: float,
) -> str:
    profile_keys = (
        "tenure", "contract", "internet_type", "monthly_charges", "payment_method",
        "service_count", "tech_support", "online_security", "online_backup",
        "device_protection", "streaming_tv", "streaming_movies",
    )  # fmt: skip
    payload = {
        "customer": {k: row.get(k) for k in profile_keys if k in row},
        "needs_attention_because": [d["text"] for d in drivers],
        "budget_usd": round(float(budget), 2),
        "menu": [
            {"id": o["id"], "description": o["customer_text"], "cost_usd": o["cost"]} for o in menu
        ],
    }
    return json.dumps(payload, default=str)


def _call_with_timeout(fn: Callable[[], str], seconds: float) -> str:
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        return future.result(timeout=seconds)
    except FutureTimeout as exc:
        raise TimeoutError(f"LLM call exceeded {seconds:.0f}s") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _parse_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    return data


def validate_llm_output(
    data: Mapping[str, Any], menu: Sequence[Offer], budget: float
) -> tuple[list[Offer], str, str, str]:
    """Return (actions, headline, customer_message, rep_notes) or raise ValueError."""
    ids = data.get("action_ids")
    if not isinstance(ids, list) or not 1 <= len(ids) <= MAX_ACTIONS:
        raise ValueError("action_ids must be a list of 1-3 ids")
    by_id = {o["id"]: o for o in menu}
    chosen: list[Offer] = []
    for action_id in ids:
        if action_id not in by_id:
            raise ValueError(f"action id not on the menu: {action_id!r}")
        if action_id in {c["id"] for c in chosen}:
            raise ValueError(f"duplicate action id: {action_id!r}")
        chosen.append(dict(by_id[action_id]))
    total = sum(c["cost"] for c in chosen)
    if total > budget + 1e-6:
        raise ValueError(f"chosen actions cost ${total:.2f} > budget ${budget:.2f}")
    for key, limit in (("headline", 120), ("customer_message", 700), ("rep_notes", 1500)):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} missing or empty")
        if len(value) > limit:
            raise ValueError(f"{key} longer than {limit} characters")
    for key in ("headline", "customer_message"):
        hit = _BANNED.search(data[key])
        if hit:
            raise ValueError(f"{key} contains banned wording: {hit.group(0)!r}")
    return (
        chosen,
        data["headline"].strip(),
        data["customer_message"].strip(),
        data["rep_notes"].strip(),
    )


def generate_offer(
    row: Mapping[str, Any],
    drivers: Sequence[Mapping[str, Any]],
    budget: float,
    provider: LLMProvider | None = None,
    context: Mapping[str, Any] | None = None,
    timeout: float = LLM_TIMEOUT_SECONDS,
) -> RetentionOffer:
    """Draft a retention offer. ``provider=None`` -> deterministic template engine.

    ``drivers`` are dicts with ``feature`` and ``text`` (strongest first).
    """
    base = template_offer(row, drivers, budget, context)
    if provider is None:
        return base

    menu = candidate_menu(row, drivers, budget)
    try:
        raw = _call_with_timeout(
            lambda: provider.generate(SYSTEM_PROMPT, build_llm_prompt(row, drivers, menu, budget)),
            timeout,
        )
        chosen, headline, message, notes = validate_llm_output(_parse_json(raw), menu, budget)
    except Exception as exc:  # any failure -> template; never surface a half-valid LLM answer
        log.warning("LLM offer rejected (%s): %s", type(exc).__name__, exc)
        base.fallback_reason = f"{provider.name}: {type(exc).__name__}: {exc}"[:300]
        base.provider = provider.name
        return base

    total = sum(a["cost"] for a in chosen)
    return RetentionOffer(
        headline=headline,
        customer_message=message,
        rep_notes=notes,
        actions=chosen,
        total_cost=round(total, 2),
        budget=float(budget),
        generated_by="LLM",
        provider=provider.name,
        model=provider.model,
        drivers=[dict(d) for d in drivers],
    )
