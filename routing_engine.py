import math
import random
import time
import os
import json
import pandas as pd
from collections import defaultdict


# =========================
# CONFIG
# =========================

DECAY_LAMBDA = 0.0000001
SHARE_PENALTY_EXPONENT = 2

MIN_PSP_SHARE = 0.01

MAX_LATENCY = 2500
MAX_COST = 3.0

# =========================
# REGIONS
# =========================

REGION_MAP = {
    # Africa
    "NGN":    "Africa",
    "KES":    "Africa",
    "GHS":    "Africa",
    "TZS":    "Africa",
    "UGX":    "Africa",
    "ZAR":    "Africa",
    "ZMW":    "Africa",
    "XAF":    "Africa",
    "XOF":    "Africa",
    "EGP":    "Africa",
    # APAC
    "IDR":    "APAC",
    "PHP":    "APAC",
    "THB":    "APAC",
    "VND":    "APAC",
    "MYR":    "APAC",
    "BDT":    "APAC",
    "JPY":    "APAC",
    "AUD":    "APAC",
    "CNY":    "APAC",
    # Europe
    "EUR":    "Europe",
    "PLN":    "Europe",
    "GBP":    "Europe",
    # LATAM
    "BRL":    "LATAM",
    "MXN":    "LATAM",
    "COP":    "LATAM",
    "ARS":    "LATAM",
    "CLP":    "LATAM",
    "PEN":    "LATAM",
    # North America
    "CAD":    "North America",
    "USD":    "North America",
    # Global
    "GLOBAL": "Global",
}


def get_country_region(country):
    return REGION_MAP.get(country, "Global")


# =========================
# TIME
# =========================

TIME_BUCKETS = ["morning", "afternoon", "evening", "night"]


def sample_time_bucket():
    return random.choice(TIME_BUCKETS)


# =========================
# LOAD PSP DATA FROM CSV
# =========================

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "payment_data.csv"
)

_df = pd.read_csv(_DATA_PATH)

PSP_COST = (
    _df.groupby("psp")["base_cost"].mean()
    .round(3)
    .to_dict()
)

PSP_LATENCY = (
    _df.groupby("psp")["latency_ms"].mean()
    .round(0)
    .astype(int)
    .to_dict()
)

_combo_psps = (
    _df.groupby(["country", "payment_method"])["psp"]
    .apply(list)
    .reset_index()
)

CONTEXTS = [
    (row["country"], row["payment_method"].upper().replace(" ", "_"), row["psp"], 1.0)
    for _, row in _combo_psps.iterrows()
]

print(f"[routing_engine] Loaded {len(PSP_COST)} PSPs, {len(CONTEXTS)} country/method contexts from payment_data.csv")


# =========================
# FAILURE MODEL
# =========================

FAILURE_COST = {
    "retryable": 0.2,
    "soft_decline": 0.5,
    "hard_fail": 1.5,
    "user_drop": 3.0
}

FAILURE_DISTRIBUTION = {
    "Quidax":  {"retryable": 0.4, "soft_decline": 0.3, "hard_fail": 0.2, "user_drop": 0.1},
    "Korapay": {"retryable": 0.3, "soft_decline": 0.3, "hard_fail": 0.25, "user_drop": 0.15},
    "Telco":   {"retryable": 0.6, "soft_decline": 0.2, "hard_fail": 0.1, "user_drop": 0.1},
}


def compute_failure_cost(psp):
    dist = FAILURE_DISTRIBUTION.get(
        psp,
        {"retryable": 0.5, "soft_decline": 0.3, "hard_fail": 0.1, "user_drop": 0.1}
    )
    return sum(dist[k] * FAILURE_COST[k] for k in dist)


# =========================
# PORTFOLIO TARGET
# =========================


# =========================
# CIRCUIT BREAKER
# =========================

from collections import deque

FAILURE_RATE_THRESHOLD = 0.85   # disable if >85% failures in last WINDOW_SIZE transactions
WINDOW_SIZE            = 20
RECOVERY_TIME          = 600    # seconds


class CircuitBreaker:

    def __init__(self):
        self.recent_outcomes = {}   # psp → deque of bool (True=success, False=failure)
        self.disabled_until  = {}   # psp → timestamp

    def _window(self, psp):
        if psp not in self.recent_outcomes:
            self.recent_outcomes[psp] = deque(maxlen=WINDOW_SIZE)
        return self.recent_outcomes[psp]

    def failure_rate(self, psp):
        window = self._window(psp)
        if not window:
            return 0.0
        return window.count(False) / len(window)

    def record_success(self, psp):
        self._window(psp).append(True)

    def record_failure(self, psp):
        self._window(psp).append(False)
        if len(self.recent_outcomes[psp]) >= 10:
            failure_rate = self.recent_outcomes[psp].count(False) / len(self.recent_outcomes[psp])
            if failure_rate > FAILURE_RATE_THRESHOLD:
                self.disabled_until[psp] = current_ts() + RECOVERY_TIME
                print(f"[CIRCUIT BREAKER] PSP {psp} DISABLED — failure rate "
                      f"{failure_rate:.0%} in last {len(self.recent_outcomes[psp])} transactions")

    def is_available(self, psp):
        until = self.disabled_until.get(psp, 0)
        if time.time() < until:
            return False
        if psp in self.disabled_until and time.time() >= until:
            del self.disabled_until[psp]
            print(f"[CIRCUIT BREAKER] PSP {psp} re-enabled for testing")
        return True

    def get_status(self):
        all_psps = set(self.recent_outcomes) | set(self.disabled_until)
        return {
            psp: {
                "failure_rate":    round(self.failure_rate(psp), 3),
                "window_size":     len(self._window(psp)),
                "disabled_until":  self.disabled_until.get(psp),
            }
            for psp in all_psps
        }


circuit_breaker = CircuitBreaker()


# =========================
# DRIFT DETECTOR
# =========================

DRIFT_THRESHOLD = 0.20  # alert if success rate drops 20% below baseline
DRIFT_MIN_WINDOW = 20   # need at least 20 transactions before detecting drift
DRIFT_WINDOW_SIZE = 50  # rolling window of last 50 outcomes


class DriftDetector:

    def __init__(self):
        self.rolling_window  = {}   # psp → deque of bool
        self.baseline_rate   = {}   # psp → float (from warm_start)
        self.already_alerted = set()

    def _window(self, psp):
        if psp not in self.rolling_window:
            self.rolling_window[psp] = deque(maxlen=DRIFT_WINDOW_SIZE)
        return self.rolling_window[psp]

    def set_baseline(self, psp, success_rate):
        self.baseline_rate[psp] = success_rate

    def record_outcome(self, psp, success):
        self._window(psp).append(success)

    def check_drift(self, psp):
        window = self._window(psp)
        if len(window) < DRIFT_MIN_WINDOW:
            return False
        baseline     = self.baseline_rate.get(psp)
        if baseline is None:
            return False
        current_rate = sum(window) / len(window)
        if baseline - current_rate > DRIFT_THRESHOLD:
            if psp not in self.already_alerted:
                self.already_alerted.add(psp)
                print(f"[DRIFT] PSP {psp} degrading — baseline {baseline:.0%} current {current_rate:.0%}")
            return True
        return False

    def get_drift_report(self):
        all_psps = set(self.rolling_window) | set(self.baseline_rate)
        report = {}
        for psp in all_psps:
            window       = self._window(psp)
            current_rate = sum(window) / len(window) if window else None
            report[psp]  = {
                "baseline_rate": self.baseline_rate.get(psp),
                "current_rate":  round(current_rate, 3) if current_rate is not None else None,
                "window_size":   len(window),
                "drift":         self.check_drift(psp),
            }
        return report


drift_detector = DriftDetector()


# =========================
# BANDIT STORE
# =========================

def current_ts():
    return time.time()


class ContextualBanditStore:
    """Local context-level bandit keyed by (country, payment_method, time_bucket)."""

    def __init__(self):
        self.context_stats = defaultdict(lambda: defaultdict(self._init))
        self.context_counts = defaultdict(lambda: defaultdict(int))

    def _init(self):
        return {"alpha": 1.0, "beta": 1.0, "ts": current_ts()}

    def _decay(self, s):
        dt = current_ts() - s["ts"]
        decay = math.exp(-DECAY_LAMBDA * dt)
        s["alpha"] *= decay
        s["beta"] *= decay
        s["ts"] = current_ts()
        s["alpha"] = max(0.01, s["alpha"])
        s["beta"]  = max(0.01, s["beta"])

    def _key(self, ctx):
        return (ctx["country"], ctx["payment_method"], ctx["time_bucket"])

    def get_stats(self, ctx, psp):
        s = self.context_stats[self._key(ctx)][psp]
        self._decay(s)
        return s

    def sample(self, ctx, psp):
        s = self.get_stats(ctx, psp)
        alpha = max(0.01, s["alpha"])
        beta  = max(0.01, s["beta"])
        return random.betavariate(alpha, beta)

    def update(self, ctx, psp, reward):
        key = self._key(ctx)
        s = self.context_stats[key][psp]
        self._decay(s)
        if reward > 0:
            s["alpha"] += 1
        else:
            s["beta"] += 1
        self.context_counts[key][psp] += 1

    def context_share(self, ctx, psp):
        key = self._key(ctx)
        total = sum(self.context_counts[key].values())
        return self.context_counts[key][psp] / total if total else 0


class RegionalBanditStore:
    """Bandit keyed by region (Africa, APAC, Europe, Americas) or 'Global'."""

    def __init__(self):
        self.stats = defaultdict(lambda: defaultdict(self._init))

    def _init(self):
        return {"alpha": 1.0, "beta": 1.0, "ts": current_ts()}

    def _decay(self, s):
        dt = current_ts() - s["ts"]
        decay = math.exp(-DECAY_LAMBDA * dt)
        s["alpha"] *= decay
        s["beta"] *= decay
        s["ts"] = current_ts()
        s["alpha"] = max(0.01, s["alpha"])
        s["beta"]  = max(0.01, s["beta"])

    def get_stats(self, key, psp):
        s = self.stats[key][psp]
        self._decay(s)
        return s

    def sample(self, key, psp):
        s = self.get_stats(key, psp)
        alpha = max(0.01, s["alpha"])
        beta  = max(0.01, s["beta"])
        return random.betavariate(alpha, beta)

    def update(self, key, psp, reward):
        s = self.stats[key][psp]
        self._decay(s)
        if reward > 0:
            s["alpha"] += 1
        else:
            s["beta"] += 1



# =========================
# ROUTER
# =========================

# =========================
# RETRY Q-LEARNING
# =========================

class RetryQLearning:
    """Q-Learning layer that controls PSP selection on retry attempts only."""
    LEARNING_RATE = 0.1
    DISCOUNT      = 0.9
    EPSILON       = 0.05   # low exploration — bad retries are costly

    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self._initialize_heuristics()

    def _initialize_heuristics(self):
        # Pre-seed Q-table with domain knowledge
        self.failure_boost = {
            "soft_decline": 0.3,    # boost premium PSPs (high approval rate)
            "retryable":    0.1,    # slight boost to similar-tier PSPs
            "hard_fail":   -0.2,   # penalize same-tier PSPs
            "user_drop":   -0.5,   # strongly avoid similar PSPs
        }

    def get_state(self, country, payment_method, attempt_number, last_failure_type):
        attempt = min(attempt_number, 3)
        return (country, payment_method, attempt, last_failure_type)

    def select_psp(self, state, available_psps, last_psp):
        candidates = [p for p in available_psps if p != last_psp]
        if not candidates:
            candidates = list(available_psps)

        if random.random() < self.EPSILON:
            return random.choice(candidates)

        failure_type = state[3]
        boost = self.failure_boost.get(failure_type, 0)

        q_values = {psp: self.q_table[state][psp] for psp in candidates}

        # Apply cost-aware heuristic boost based on failure type
        for psp in candidates:
            cost = PSP_COST.get(psp, 1.5)
            if failure_type == "soft_decline" and cost > 2.0:
                q_values[psp] += boost
            elif failure_type == "retryable" and cost < 1.5:
                q_values[psp] += abs(boost)
            elif failure_type in ["hard_fail", "user_drop"]:
                q_values[psp] += boost

        return max(q_values, key=q_values.get)

    def update(self, state, psp, reward, next_state, done):
        current_q = self.q_table[state][psp]
        if done:
            target = reward
        else:
            next_max_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
            target = reward + self.DISCOUNT * next_max_q
        self.q_table[state][psp] += self.LEARNING_RATE * (target - current_q)

    def save_state(self):
        return {str(k): dict(v) for k, v in self.q_table.items()}

    def load_state(self, data):
        for key_str, psps in data.items():
            try:
                key = eval(key_str)
            except Exception:
                key = key_str
            for psp, q_val in psps.items():
                self.q_table[key][psp] = q_val


retry_ql = RetryQLearning()


class Router:

    def __init__(self, local_bandit, regional_bandit, global_bandit):
        self.local = local_bandit
        self.regional = regional_bandit
        self.global_ = global_bandit
        self.max_share = 0.6

    def route(self, ctx, psps, amount=500):
        decisions = []
        region = get_country_region(ctx["country"])

        # Cache local stats once to avoid double-applying decay
        local_stats = {psp: self.local.get_stats(ctx, psp) for psp in psps}
        total_trials = sum(s["alpha"] + s["beta"] for s in local_stats.values())

        for psp in psps:
            if not circuit_breaker.is_available(psp):
                continue

            # Hard constraints — eliminate PSP before scoring
            if PSP_LATENCY.get(psp, MAX_LATENCY) > 2000:
                continue
            if amount < 100 and PSP_COST.get(psp, MAX_COST) > 2.0:
                continue

            s = local_stats[psp]
            alpha, beta = max(0.01, s["alpha"]), max(0.01, s["beta"])

            # Thompson samples from all three levels
            local_sample    = random.betavariate(alpha, beta)
            regional_sample = self.regional.sample(region, psp)
            global_sample   = self.global_.sample("global", psp)

            # UCB bonus from local stats
            psp_trials   = alpha + beta
            success_rate = alpha / psp_trials
            if total_trials > 0 and psp_trials > 0:
                ucb_bonus = math.sqrt(0.5 * math.log(total_trials) / psp_trials)
            else:
                ucb_bonus = 1.0
            ucb_score = success_rate + ucb_bonus

            # Hierarchical combination (with UCB)
            combined = 0.5 * local_sample + 0.3 * regional_sample + 0.2 * global_sample
            combined = 0.9 * combined + 0.1 * ucb_score

            # Cost and latency scores (higher = cheaper/faster = better)
            cost_score    = 1 - (PSP_COST.get(psp, MAX_COST) / MAX_COST)
            latency_score = 1 - (PSP_LATENCY.get(psp, MAX_LATENCY) / MAX_LATENCY)

            bandit_cost_score = (0.8 * combined) + (0.1 * cost_score) + (0.1 * latency_score)

            final_score = bandit_cost_score

            # Minimum traffic floor: boost under-explored PSPs to keep all warm
            current_share = bandit_local.context_share(ctx, psp)
            if current_share < MIN_PSP_SHARE:
                final_score += (MIN_PSP_SHARE - current_share) * 0.5

            decisions.append({"psp": psp, "score": final_score})

        if not decisions:
            # All PSPs eliminated — fall back to lowest-latency PSP ignoring constraints
            fallback = min(psps, key=lambda p: PSP_LATENCY.get(p, MAX_LATENCY))
            return fallback

        decisions.sort(key=lambda x: x["score"], reverse=True)
        return decisions[0]["psp"]


# =========================
# PRODUCTION INTERFACE
# =========================

bandit_local    = ContextualBanditStore()
bandit_regional = RegionalBanditStore()
bandit_global   = RegionalBanditStore()  # keyed by "global"
router_global   = Router(bandit_local, bandit_regional, bandit_global)


def warm_start(df):
    """Pre-populate all three bandit levels from payment_data.csv success_rate."""
    for _, row in df.iterrows():
        country        = row["country"]
        payment_method = row["payment_method"]
        psp            = row["psp"]
        sr             = row["success_rate"]
        alpha          = sr * 100
        beta           = (1 - sr) * 100
        region         = get_country_region(country)

        # Local
        ctx = {"country": country, "payment_method": payment_method, "time_bucket": "afternoon"}
        key = bandit_local._key(ctx)
        s = bandit_local.context_stats[key][psp]
        s["alpha"], s["beta"], s["ts"] = alpha, beta, current_ts()

        # Regional
        s = bandit_regional.stats[region][psp]
        s["alpha"], s["beta"], s["ts"] = alpha, beta, current_ts()

        # Global
        s = bandit_global.stats["global"][psp]
        s["alpha"], s["beta"], s["ts"] = alpha, beta, current_ts()

        # Drift detector baseline
        drift_detector.set_baseline(psp, sr)

    print(f"[routing_engine] Warm-started all three bandit levels with {len(df)} PSP records")


# =========================
# PERSIST BANDIT STATE
# =========================

_BANDIT_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bandit_state.json")


def _serialise_bandit(store_stats):
    return {
        str(key): {
            psp: {"alpha": s["alpha"], "beta": s["beta"], "ts": s["ts"]}
            for psp, s in psps.items()
        }
        for key, psps in store_stats.items()
    }


def _deserialise_into(store_stats, data):
    for key_str, psps in data.items():
        try:
            key = eval(key_str)
        except:
            key = key_str
        for psp, s in psps.items():
            store_stats[key][psp] = {
                "alpha": max(0.01, min(1000, float(s.get("alpha", 1.0)))),
                "beta":  max(0.01, min(1000, float(s.get("beta",  1.0)))),
                "ts":    current_ts(),
            }


def save_bandit_state():
    state = {
        "local":    _serialise_bandit(bandit_local.context_stats),
        "regional": _serialise_bandit(bandit_regional.stats),
        "global":   _serialise_bandit(bandit_global.stats),
        "local_counts": {
            str(k): dict(v) for k, v in bandit_local.context_counts.items()
        },
        "retry_ql": retry_ql.save_state(),
    }
    with open(_BANDIT_STATE_PATH, "w") as f:
        json.dump(state, f)


def load_bandit_state():
    with open(_BANDIT_STATE_PATH) as f:
        state = json.load(f)

    _deserialise_into(bandit_local.context_stats, state["local"])
    _deserialise_into(bandit_regional.stats,      state["regional"])
    _deserialise_into(bandit_global.stats,        state["global"])

    for key_str, psps in state.get("local_counts", {}).items():
        key = eval(key_str)
        for psp, count in psps.items():
            bandit_local.context_counts[key][psp] = count

    if "retry_ql" in state:
        retry_ql.load_state(state["retry_ql"])

    print(f"[routing_engine] Loaded bandit state from bandit_state.json "
          f"({len(state['local'])} local, {len(state['regional'])} regional, {len(state['global'])} global contexts)")


if os.path.exists(_BANDIT_STATE_PATH):
    load_bandit_state()
else:
    warm_start(_df)


# =========================
# ROUTING FUNCTIONS
# =========================

def route_transaction(transaction):
    ctx = {
        "country":        transaction["country"],
        "payment_method": transaction["payment_method"],
        "time_bucket":    transaction.get("time_bucket", "afternoon"),
    }

    psps = None
    for c, m, p, _ in CONTEXTS:
        if c == ctx["country"] and m == ctx["payment_method"]:
            psps = p
            break

    if not psps:
        return None

    return router_global.route(ctx, psps, amount=transaction.get("amount", 500))


router = router_global


def update_bandit(context, psp, reward):
    ctx = {
        "country":        context[0],
        "payment_method": context[1],
        "time_bucket":    "afternoon",
    }
    region  = get_country_region(context[0])

    bandit_local.update(ctx, psp, reward)
    bandit_regional.update(region, psp, reward)
    bandit_global.update("global", psp, reward)

    if reward > 0:
        circuit_breaker.record_success(psp)
    else:
        circuit_breaker.record_failure(psp)

    drift_detector.record_outcome(psp, reward > 0)
    drift_detector.check_drift(psp)

    save_bandit_state()


def route_transaction_with_trace(transaction):
    selected_psp = route_transaction(transaction)

    ctx = {
        "country":        transaction["country"],
        "payment_method": transaction["payment_method"],
        "time_bucket":    transaction.get("time_bucket", "afternoon"),
    }

    psps = None
    for c, m, p, _ in CONTEXTS:
        if c == ctx["country"] and m == ctx["payment_method"]:
            psps = p
            break

    if not psps:
        return selected_psp, {"error": "no_psps_found"}

    region = get_country_region(ctx["country"])

    # Compute total_trials across all PSPs in this context (matches route())
    local_stats_cache = {psp: bandit_local.get_stats(ctx, psp) for psp in psps}
    total_trials = sum(s["alpha"] + s["beta"] for s in local_stats_cache.values())

    raw_scores = {
        psp: s["alpha"] / (s["alpha"] + s["beta"])
        for psp, s in local_stats_cache.items()
    }

    decisions = []

    for psp in psps:
        local_s = local_stats_cache[psp]
        reg_s   = bandit_regional.get_stats(region, psp)
        glob_s  = bandit_global.get_stats("global", psp)

        local_sample    = random.betavariate(max(0.01, local_s["alpha"]), max(0.01, local_s["beta"]))
        regional_sample = random.betavariate(max(0.01, reg_s["alpha"]),   max(0.01, reg_s["beta"]))
        global_sample   = random.betavariate(max(0.01, glob_s["alpha"]),  max(0.01, glob_s["beta"]))

        hierarchical = 0.5 * local_sample + 0.3 * regional_sample + 0.2 * global_sample

        psp_trials   = local_s["alpha"] + local_s["beta"]
        success_rate = local_s["alpha"] / psp_trials
        if total_trials > 0 and psp_trials > 0:
            ucb_bonus = math.sqrt(0.5 * math.log(total_trials) / psp_trials)
        else:
            ucb_bonus = 1.0
        ucb_score = success_rate + ucb_bonus

        combined          = 0.9 * hierarchical + 0.1 * ucb_score
        cost_score        = 1 - (PSP_COST.get(psp, MAX_COST) / MAX_COST)
        latency_score     = 1 - (PSP_LATENCY.get(psp, MAX_LATENCY) / MAX_LATENCY)
        bandit_cost_score = 0.8 * combined + 0.1 * cost_score + 0.1 * latency_score
        final_score       = bandit_cost_score

        decisions.append({
            "psp":               psp,
            "score":             round(final_score, 4),
            "local_sample":      round(local_sample, 4),
            "regional_sample":   round(regional_sample, 4),
            "global_sample":     round(global_sample, 4),
            "hierarchical":      round(hierarchical, 4),
            "ucb_bonus":         round(ucb_bonus, 4),
            "ucb_score":         round(ucb_score, 4),
            "cost_score":        round(cost_score, 4),
            "latency_score":     round(latency_score, 4),
            "bandit_cost_score": round(bandit_cost_score, 4),
            "final_score":       round(final_score, 4),
        })

    decisions.sort(key=lambda x: x["final_score"], reverse=True)
    best_psp = decisions[0]["psp"] if decisions else None

    # Plain-English winner explanation
    if best_psp:
        psp_sr      = round(raw_scores.get(best_psp, 0) * 100, 1)
        psp_cost    = PSP_COST.get(best_psp, MAX_COST)
        psp_latency = PSP_LATENCY.get(best_psp, MAX_LATENCY)
        why_winner  = (
            f"Strong success history ({psp_sr}%) + "
            f"low cost (${psp_cost:.2f}) + "
            f"fast latency ({psp_latency:,}ms)"
        )
    else:
        why_winner = "No PSPs available"

    trace = {
        "context":      ctx,
        "region":       region,
        "selected_psp": selected_psp,
        "best_psp":     best_psp,
        "psp_ranking":  decisions,
        "why_winner":   why_winner,
        "reason": (
            "selected optimal PSP"
            if selected_psp == best_psp
            else "selected due to exploration / constraints / portfolio shaping"
        ),
    }

    return selected_psp, trace
