# ZoneTrend — Supply/Demand Zone Detection: Deep Code Review

**Scope:** `src/zones/zone_detector.py` (1045 lines), `config/zone_config.yaml`, with supporting
context from `src/data/preprocessor.py`.
**Focus:** algorithm correctness, market logic, and **detection completeness** — not coding style.
**Empirical basis:** every quantitative claim below was produced by running the *actual* module
functions on the real processed dataset `data/processed/IDX_NSEI.csv` (2,812 daily candles,
2015‑01‑02 → 2026‑..., ^NSEI / Nifty 50).

> **Assumption note.** The brief refers to four patterns "RDB, DBR, RBR, DBD". I read "RDB" as
> **RBD** (Rally‑Base‑Drop), which is the term the code and config use. Where the brief mentions
> "merge logic", note that **no zone‑merge step exists in the code** — the only overlap suppression
> is the base‑scan `i = j` jump (analysed in §3.4). I call that out explicitly rather than reviewing
> a function that isn't there.

---

## 0. Headline conclusion

**The detector is severely under‑detecting. It finds ~44 zones across ~11 years (~3.85 zones/year)
on Nifty daily. The realistic, quality‑controlled target is ~150–220 (~3.5–5×).**

Two findings dominate everything else:

1. **Only 2 of the 4 supply/demand patterns are implemented.** The code detects **DBR** (demand) and
   **RBD** (supply) — the two *reversal* patterns — and is **structurally incapable** of detecting the
   two *continuation* patterns **RBR** and **DBD**, because it hard‑couples the arrival‑leg direction
   to the zone type (§3.1). On this dataset continuation patterns represent **+93%** more zones
   (44 → 85). This is the single largest gap and it is a *design* limitation, not a tuning issue.

2. **The departure test is the most aggressive filter in the pipeline, and it measures the wrong
   thing.** It requires a *single* candle, *immediately* after the base, with body ≥ 0.9×ATR. That
   rejects **576 of 793** direction‑correct departures (73%). Real institutional "leg‑outs" are
   frequently 2–3 candles; a one‑candle body test cannot see them (§3.2).

The measured detection funnel (§2) shows where every candidate dies. The ranked fixes (§6) and the
top‑10 table (§7) are all backed by re‑running the pipeline with each change applied in isolation and
in combination.

---

## 1. What the pipeline actually does (function‑by‑function)

The public entry point is `detect(df, zcfg, logger)` (lines 795‑964). It runs seven stages.

### 1.1 `_is_base_candle(...)` — lines 208‑228
Pure per‑candle test. Returns `True` when **both**:

- `(High − Low) ≤ base_range_multiplier × ATR`  (range tightness)
- `|Close − Open| ≤ base_body_multiplier × ATR`  (body tightness)

Returns `False` if ATR is NaN/≤0 (so the first 14 candles, during ATR warm‑up, can never be bases).
Config: `base_range_multiplier = 1.8`, `base_body_multiplier = 0.5`.

### 1.2 `_find_bases(df, zcfg)` — lines 231‑302
Single forward scan. Starting at each base‑eligible candle it **greedily extends** a run while the
next candle is also base‑eligible **and** the run length `< max_base_length` (3). For each maximal run
of length ≥ `min_base_length` (1) it computes `top = max(High)`, `bottom = min(Low)`,
`avg_atr = nanmean(ATR)`, and applies a **total‑range gate**: `top − bottom ≤ base_range_multiplier ×
avg_atr` (line 287). If the gate passes the base is recorded. Then — **regardless of whether the gate
passed** — it jumps the cursor with `i = j` (line 298), skipping the entire run. (Bug: §3.4.)

### 1.3 `_check_departure(...)` — lines 307‑377
Looks at **exactly one** candle: `dep_idx = base_end_idx + 1` (line 337). Requires:

- **Direction** — demand ⇒ `Close ≥ Open` (bullish); supply ⇒ `Close < Open` (bearish).
- **Body size** — `|Close − Open| / avg_atr ≥ departure_strength` (0.9). Note: normalised by the
  *base's* avg ATR, not the departure candle's own ATR — a defensible choice.
- **Close ratio** — demand ⇒ `(Close − Low)/range ≥ departure_close_ratio` (0.3); supply ⇒
  `(High − Close)/range ≥ 0.3`.

### 1.4 `_check_arrival(...)` — lines 382‑449
Examines the `arrival_lookback` (4) candles **before** the base (`[base_start−4, base_start−1]`, base
excluded). Requires **two** things, both hard gates:

- **Net move** — `Close[base_start−1] − Close[base_start−4]` must be negative (demand) / positive
  (supply) **and** `|net|/avg_atr ≥ arrival_min_move` (0.5).
- **Majority direction** — a *strict* majority of the 4 arrival candles must be bearish (demand) /
  bullish (supply): `count > n/2`.

**This is where the zone type is hard‑wired to the arrival direction** — the root cause of the missing
continuation patterns (§3.1).

### 1.5 `_score_zone(...)` — lines 600‑689
Weighted 0–1 score from five normalised components: departure strength (0.40, itself 70% body /
30% close‑ratio), base tightness (0.25), arrival momentum (0.20), and a split volume term (0.15,
divided between departure volume and *base* volume per the `base_volume` section). Clean — uses only
formation‑time information.

### 1.6 `_track_zone_status(...)` — lines 694‑790
Walks forward from `formation_idx + 1`. Marks `invalid` on a close (or wick) beyond the boundary;
increments `test_count` on each fresh re‑entry; marks `consumed` at `max_test_count` (5). Then
computes `freshness_score` from the *final* `test_count` via `decay_values` (Priority 2). **This stage
deliberately uses future data** — fine for post‑hoc study, but it contaminates `adjusted_strength`
(§3.7).

### 1.7 `_add_weekly_confirmation(...)` — lines 491‑568 (+ `_resample_to_weekly` 454‑488)
Resamples daily→weekly (W‑FRI), **recursively calls `detect()`** on the weekly frame, and flags any
daily zone that overlaps a same‑type weekly zone formed on/before it. Behaviour analysed in §3.5.

### 1.8 `detect(...)` — lines 795‑964
Orchestrates the above. For each base it tries **both** `("demand", "supply")` (line 853); the two are
mutually exclusive (a candle can't be both bullish and bearish), so a base yields at most one zone.
`adjusted_strength = min(strength × freshness + weekly_bonus, 1.0)` (lines 948‑953).

---

## 2. The detection funnel (measured on real data)

Running the *actual* functions on `IDX_NSEI.csv`:

| Stage | Count | Notes |
|---|---:|---|
| Candles | 2,812 | |
| Per‑candle base‑eligible | 1,730 (61.5%) | `base_range_multiplier=1.8` is permissive at candle level |
| Maximal runs (len ≥ 1) | 859 | |
| Runs passing total‑range gate | 794 | accepted bases |
| **Runs FAILING total‑range gate** | **65** | **182 candles swallowed by `i = j` and never re‑examined** |
| Departure direction‑correct (base×type) | 793 | almost every base has a directional next candle |
| → also body ≥ 0.9×ATR | **217** | **576 killed by body size** |
| → also close‑ratio ≥ 0.3 | 217 | **0 killed by close‑ratio → this threshold is currently inert** |
| Arrival pass (net move) | 87 | **130 killed by net‑move** |
| → also majority direction | **44** | **43 more killed by majority rule** |
| **Final zones** | **44** | demand 23 / supply 21 |

**Status of the 44 detected zones:** `invalid` 38, `consumed` 3, `tested` 3, **`active` 0**. Over an
11‑year window it's expected that old zones eventually get cut, but **zero active zones** combined with
the tiny count means the detector currently produces almost nothing usable at the right edge.

**Restrictiveness ranking (existing 2 patterns):**
1. Departure body size — 576 rejections (by far the largest).
2. Arrival net‑move — 130 rejections.
3. Arrival majority‑direction — 43 rejections.
4. Base total‑range gate + `i=j` skip — 65 runs / 182 candles lost.
5. Departure close‑ratio — 0 rejections (mis‑tuned, see §3.6).

---

## 3. Bugs and logic issues

### 3.1 🔴 CRITICAL — Continuation patterns (RBR, DBD) are impossible to detect
**Where:** `_check_arrival` (lines 427‑447) coupled with `detect`'s type loop (line 853).

Zone polarity in S/D theory is set by the **leg‑out (departure) direction**: a strong departure *up*
from a base ⇒ **demand** (support); a strong departure *down* ⇒ **supply** (resistance) — *regardless*
of how price arrived. The arrival leg only distinguishes **reversal** (DBR/RBD) from **continuation**
(RBR/DBD).

The code instead forces:

```python
# demand REQUIRES a downward arrival; supply REQUIRES an upward arrival
if zone_type == "demand":
    net_ok = (net_move < 0) and (net_move_atr >= min_move)
else:
    net_ok = (net_move > 0) and (net_move_atr >= min_move)
```

So a base with an **upward** arrival and an **upward** departure (textbook **RBR** demand‑continuation)
is rejected: the bullish departure tags it `demand`, but the demand branch demands a *negative*
arrival. Symmetrically **DBD** is impossible.

**Measured impact:** there are **21 RBR** and **50 DBD** structurally valid candidates. A correct
4‑pattern implementation yields **85 zones vs 44 (+93%)** with *no other change*. This is the highest‑
value fix and it is purely structural.

### 3.2 🔴 CRITICAL — Departure measured as one candle, immediately adjacent
**Where:** `_check_departure`, `dep_idx = base_end_idx + 1` (line 337); single‑candle body test
(lines 363‑367).

Two compounding problems:

- **One candle only.** Institutional leg‑outs are routinely 2–3 candles (a moderate breakout candle
  then an expansion candle, or a small inside candle the day after the base then the real thrust). The
  code sees only `base_end+1`.
- **Body ≥ 0.9×ATR is a very high bar for a single candle.** ATR ≈ average *range*, and a candle's
  *body* is always ≤ its range, typically 0.5–0.7×ATR even for strong candles. Requiring body alone to
  exceed 0.9×ATR means only near‑marubozu candles qualify — hence the 73% kill rate (576/793).

Interaction with §3.3: because the base scan is greedy, a small candle that is really the *first*
leg‑out candle can be absorbed into the base, pushing the true thrust to `base_end+2`, which is never
inspected.

**Measured impact (isolated, 2 patterns):** lowering the single‑candle threshold 0.9 → 0.7 gives 80
zones (+82%); → 0.6 gives 96 (+118%). A multi‑candle leg‑out test (close displaced ≥1.0×ATR beyond the
base within 3 candles) gives 56 (+27%) but captures a *different, more faithful* set (genuine
multi‑bar legs) — best combined with a modestly lower per‑candle floor. See §6.2.

### 3.3 🟠 Greedy base grouping can misplace or shrink the base
**Where:** `_find_bases` extension loop (lines 268‑278).

The run always starts at the *first* eligible candle and stops at `max_base_length`. Consider four
consecutive base‑like candles followed by the thrust: the first group becomes candles [0,1,2], the
"departure" is checked at candle 3 (still a base candle → fails), the cursor jumps, and the leftover
[3] is re‑grouped as a 1‑candle base whose departure (candle 4) *does* fire — but the zone boundary is
now drawn from a single candle instead of the true consolidation. **Base boundaries (and therefore
zone width, midpoint, and every downstream level) become a function of grid alignment, not market
structure.** A windowed search (try each base length and keep the variant that yields the strongest
departure) removes this dependence (§6.4).

### 3.4 🟠 Total‑range gate + `i = j` swallows candles (off‑by‑design skip)
**Where:** lines 287‑298.

```python
if avg_atr > 0 and total_range <= range_mult * avg_atr:
    bases.append({...})
# Advance past this base — runs even when the gate FAILED
i = j
```

When several candles are *individually* tight but *collectively* span more than 1.8×avg_atr
(stair‑stepping), the gate fails — and the cursor still jumps past **all** of them. A valid tighter
sub‑base (e.g. just the last candle before the thrust) is never tried. **65 runs / 182 candles** are
discarded this way.

**Honest sizing:** at *current* thresholds, fixing this adds 40 candidate bases (794 → 834) but **~0
net zones**, because those particular sub‑bases rarely have a qualifying single‑candle departure. It is
a genuine correctness bug whose payoff **compounds once §3.1/§3.2 are relaxed** — leave it in and you
re‑introduce a silent loss as soon as the departure test is loosened.

### 3.5 🟠 Weekly confirmation recurses ~44 levels deep (fragile + wasteful), but does *not* crash
**Where:** `_add_weekly_confirmation` line 521 calls `detect(weekly_df, zcfg)`, and `detect`
unconditionally calls `_add_weekly_confirmation` again.

I initially expected infinite recursion. **Tested — it terminates.** Each weekly resample + 14‑period
ATR warm‑up drops ~13 rows, so the chain detect(daily)→detect(weekly)→detect(weekly²)→… shrinks until
the frame is too short, ~44 levels deep, then unwinds. Top‑level flags *are* computed (3 zones
weekly‑confirmed here). So it is **not silently broken** — but it runs the full pipeline (including
forward status tracking) dozens of times per call, computing meaningless "weekly‑of‑weekly‑of‑weekly"
zones, and it is one `min_periods` change away from blowing the stack. Add an `enable_weekly` guard
(§6.5). (Whole `detect` currently ≈ 0.66 s, so this is robustness/clarity, not a present‑day speed
emergency.)

### 3.6 🟡 `departure_close_ratio = 0.3` is inert; config values contradict their own docs
**Where:** config + `_check_departure`.

Because a candle that already has body ≥ 0.9×ATR necessarily closes near its extreme, the 0.3
close‑ratio gate rejects **0** candidates. It does nothing. The config comments also disagree with the
values: `base_range_multiplier` is 1.8 with a comment saying "Typical range 0.5–1.5 … range > 1×ATR is
too large to be a base"; `departure_strength` is 0.9 with a comment describing "1.2"; arrival comments
say "5 candles"/"1.5×ATR" while the values are 4/0.5. Not bugs per se, but the knobs no longer match
the stated philosophy, which makes tuning misleading.

### 3.7 🟡 `adjusted_strength` embeds look‑ahead — dangerous as an ML feature
**Where:** lines 944‑953; `freshness_score` from final `test_count` (lines 780‑788).

Detection itself is correctly **free of look‑ahead** (a zone is "known" only at the departure close —
good). But `adjusted_strength = strength × freshness + weekly_bonus`, and `freshness` is derived from
the *full‑history* test count. The docstring even recommends `adjusted_strength` "for ranking and
filtering." If that column is fed to a model as a feature at time *t*, it leaks the future. Keep raw
`strength` as the only formation‑time score and clearly quarantine `adjusted_strength` as
analysis‑only (§6.6). Per the project's own data‑science standards (leakage prevention), this matters.

### 3.8 🟡 Volume‑driven scoring is unreliable on the primary instrument
`^NSEI` is a spot index: Yahoo "volume" is a proxy, with **29 zero‑volume days** and 19 NaN
`VolumeRatio` rows here. Yet volume drives 25% of the score (`volume_weight` 0.15 + `base_volume`
0.10). For indices this injects noise into ranking; for the cash equities (`RELIANCE.NS`) it's fine.
Make the volume contribution degrade to neutral and re‑normalise the remaining weights when volume is
absent/unreliable (§6.8).

### 3.9 🟢 Minor correctness/consistency
- **Asymmetric direction test:** demand departure uses `Close ≥ Open` while supply uses `Close < Open`;
  arrival majority uses `< Open` (bearish, strict) vs `≥ Open` (bullish, inclusive). Dojis are counted
  as bullish/up. Negligible but inconsistent.
- **Outliers not excluded:** `IsOutlier` is computed in preprocessing but never used; a gap/outlier
  candle can serve as a "departure."
- **`zone_num` logging:** "before strength filter" and "after" are always equal (the counter only
  increments on append, and `min_strength_threshold = 0.0`), so the log line is misleading.
- **Last weekly bar** can be a partial week treated as complete (minor right‑edge effect).

---

## 4. Coverage evaluation — is it detecting too few zones? (Yes)

**~44 zones / 11 years ≈ 3.85/year** on a liquid daily index. Discretionary S/D traders typically
mark far more (tens of live/historic zones). The funnel (§2) shows the count is throttled by three
multiplicative gates that each remove the majority of survivors, *plus* a structural halving from the
missing patterns:

- **Structural halving:** continuation patterns excluded → ceiling cut ~50% (§3.1).
- **Departure gate:** removes 73% of direction‑correct candidates (§3.2).
- **Arrival gate (two hard sub‑tests):** removes ~80% of departure survivors (§2).

Because the gates are *multiplicative*, the survival rate is roughly `0.5 (patterns) × 0.27 (departure)
× 0.20 (arrival) ≈ 2.7%` of bases — matching the observed 44/794 ≈ 5.5% within the detected half.

**Specific code locations where valid zones are silently lost:**
- Lines 427‑431 — every RBR/DBD base.
- Line 337 / 363‑367 — every multi‑candle leg‑out, and every base whose thrust is at `base_end+2`.
- Line 298 — every tight sub‑base inside a run that fails the aggregate range gate.
- Lines 442‑447 — clean drops/rallies into a base that happen to have ≤ 50% in‑direction candles
  (e.g. a sharp 1‑candle plunge then 3 small inside days).

---

## 5. False negatives and false positives

### 5.1 False negatives (valid zones missed) — the dominant failure mode
1. **All RBR / DBD continuation zones** (~71 candidates). *Most important.*
2. **Multi‑candle leg‑outs** failing the one‑candle 0.9×ATR body test (576 direction‑correct
   departures rejected; a large fraction are real 2–3 candle thrusts).
3. **Thrust at `base_end+2`** after a small post‑base candle — never inspected.
4. **Tight sub‑bases** inside runs that fail the aggregate range gate (§3.4).
5. **V‑shaped / single‑bar approaches** that violate the 4‑candle net‑move or majority rule (§3.2/§2).

### 5.2 False positives (poor zones accepted)
1. **Wide single‑candle "bases":** `base_range_multiplier = 1.8` lets a single candle with range up to
   1.8×ATR (a long‑legged, high‑volatility indecision bar) count as a base, producing a ~1.8‑ATR‑tall
   zone — wide, low‑precision, poor R:R. This *contradicts the config's own "≤1×ATR" philosophy.*
2. **Counter‑trend zones** with no higher‑timeframe/trend context (only the weekly *bonus* exists, and
   it's additive, never a filter) — fading a strong trend at a freshly minted zone is low‑probability.
3. **Volume‑inflated/deflated scores** on index data (§3.8).
4. **Outlier‑candle departures** (gaps) accepted as institutional thrusts (§3.9).

---

## 6. Ranked improvements (reasoning · impact · code)

Impact figures are measured on `IDX_NSEI.csv`; "isolated" = that change alone vs the current 44.

### 6.1 Detect all four patterns — decouple polarity from arrival  ·  +93% (44→85)
**Reasoning:** polarity is the departure's job; arrival only labels reversal vs continuation.
**Impact:** +41 zones isolated; the structural ceiling roughly doubles.

```python
def _check_arrival(df, base_start_idx, avg_atr, lookback, min_move):
    """Return (is_directional, move_atr, direction) — direction in {'up','down',None}."""
    start_idx = max(0, base_start_idx - lookback)
    end_idx   = base_start_idx
    if end_idx - start_idx < 1 or avg_atr <= 0:
        return False, 0.0, None
    net = float(df.iloc[end_idx-1]["Close"]) - float(df.iloc[start_idx]["Close"])
    move_atr = abs(net) / avg_atr
    if move_atr < min_move:
        return False, round(move_atr, 4), None
    return True, round(move_atr, 4), ("up" if net > 0 else "down")

# in detect(), per base:
up_ok,  dep_up = _check_departure(df, base_end, "demand", avg_atr, dep_strength, dep_cr)   # bullish thrust
dn_ok,  dep_dn = _check_departure(df, base_end, "supply", avg_atr, dep_strength, dep_cr)   # bearish thrust
if up_ok:   zone_type, dep_info, dep_dir = "demand", dep_up, "up"
elif dn_ok: zone_type, dep_info, dep_dir = "supply", dep_dn, "down"
else:       continue

arr_ok, arr_move_atr, arr_dir = _check_arrival(df, base_start, avg_atr, lookback, min_move)
if not arr_ok:
    continue

# Classify (store as a column; do NOT use it to gate polarity)
if zone_type == "demand":
    structure = "DBR" if arr_dir == "down" else "RBR"   # reversal vs continuation
else:
    structure = "RBD" if arr_dir == "up"   else "DBD"
```

> Optional: keep a `require_reversal` flag if you ever want the old behaviour, but default to all four.

### 6.2 Fix the departure test — multi‑candle leg‑out + sane floor  ·  +27% to +118% isolated
**Reasoning:** capture genuine 2–3 candle thrusts; stop equating "strong" with "one marubozu".
**Impact:** isolated 0.9→0.6 single‑candle = 96 (+118%); leg‑out(K=3, ≥1.0×ATR displacement) = 56
(+27%) but a *truer* set. Best: require the first post‑base candle to close in‑direction **and** the
close to be displaced ≥ `leg_disp×ATR` beyond the base edge within `leg_max` candles.

```python
def _check_departure(df, base_end_idx, zone_type, base_top, base_bottom, avg_atr,
                     dep_strength, dep_close_ratio, leg_max=3, leg_disp=1.0):
    n = len(df)
    first = base_end_idx + 1
    if first >= n or avg_atr <= 0:
        return False, None
    r = df.iloc[first]; o, c, h, l = map(float, (r.Open, r.Close, r.High, r.Low))
    rng = h - l
    if rng < 1e-10:
        return False, None
    # 1) first leg-out candle closes in-direction with a real (not huge) body
    if zone_type == "demand":
        if not (c >= o and (c - l)/rng >= dep_close_ratio):           # bullish, closed up
            return False, None
    else:
        if not (c <  o and (h - c)/rng >= dep_close_ratio):           # bearish, closed down
            return False, None
    if abs(c - o)/avg_atr < dep_strength:                              # lower this floor to ~0.5–0.6
        # allow a soft first candle IF the multi-candle leg confirms below
        pass
    # 2) cumulative displacement of close beyond the base within leg_max candles
    best = 0.0
    for k in range(1, leg_max + 1):
        di = base_end_idx + k
        if di >= n:
            break
        close = float(df.iloc[di]["Close"])
        disp = (close - base_top)/avg_atr if zone_type == "demand" else (base_bottom - close)/avg_atr
        best = max(best, disp)
    if best < leg_disp:
        return False, None
    return True, {"dep_idx": first, "dep_body_atr": round(abs(c-o)/avg_atr,4),
                  "dep_close_ratio": round(((c-l) if zone_type=="demand" else (h-c))/rng,4),
                  "dep_leg_atr": round(best,4),
                  "dep_volume_ratio": float(r.get("VolumeRatio", np.nan))}
```

Add `departure_leg_max: 3` and `departure_leg_disp: 1.0` to config; treat `departure_strength` as a
*soft* per‑candle floor (~0.5–0.6) rather than the sole gate.

### 6.3 Make the arrival a soft requirement (net‑move only) or a score  ·  +98% isolated
**Reasoning:** the majority‑direction rule throws away clean approaches; the arrival is *context*,
not the imbalance evidence. Keep a light net‑move floor to ensure a real leg‑in; drop the majority
gate (or move it into scoring).
**Impact:** net‑only = 87 (+98%) isolated; removing arrival entirely = 217 (too loose → use net‑only).
With §6.1+§6.2 combined, net‑only arrival reaches **214 (≈4.9×)**.

```python
# Hard gate: just a directional leg-in of sufficient size (either direction; §6.1 classifies it)
arr_ok, arr_move_atr, arr_dir = _check_arrival(df, base_start, avg_atr, lookback, min_move)
if not arr_ok:
    continue
# Soft: fold "clean approach" into the score instead of rejecting
arrival_cleanliness = _directional_fraction(df, base_start, lookback, arr_dir)  # 0..1, used in _score_zone
```

### 6.4 Replace greedy runs with a windowed base search  ·  correctness (boundary accuracy) + recall
**Reasoning:** removes grid‑alignment dependence (§3.3) and the swallow bug (§3.4) together; lets the
*best* base/length win.

```python
def _find_bases(df, zcfg):
    rm, bm = zcfg["base_range_multiplier"], zcfg["base_body_multiplier"]
    minl, maxl = zcfg["min_base_length"], zcfg["max_base_length"]
    H,L,O,C,Aatr = (df[x].values for x in ("High","Low","Open","Close","ATR"))
    n = len(df); bases = []
    for end in range(n):                       # anchor on the LAST base candle (next candle is the thrust)
        if not _is_base_candle(H[end],L[end],O[end],C[end],Aatr[end],rm,bm):
            continue
        for length in range(minl, maxl+1):     # try every admissible base length ending at `end`
            start = end - length + 1
            if start < 0:
                break
            idx = range(start, end+1)
            if not all(_is_base_candle(H[k],L[k],O[k],C[k],Aatr[k],rm,bm) for k in idx):
                break
            top, bottom = float(H[start:end+1].max()), float(L[start:end+1].min())
            avg_atr = float(np.nanmean(Aatr[start:end+1]))
            if avg_atr > 0 and (top-bottom) <= rm*avg_atr:
                bases.append({"start_idx":start,"end_idx":end,"length":length,
                              "top":top,"bottom":bottom,"avg_atr":avg_atr})
    return bases     # de-dup / keep strongest-departure variant per `end` in detect()
```

De‑duplicate downstream (e.g. per `end` index keep the variant with the strongest departure leg) so a
single thrust doesn't spawn three near‑identical zones.

### 6.5 Guard the weekly recursion  ·  robustness/perf (no coverage change)
```python
def detect(df, zcfg, logger=None, enable_weekly=True):
    ...
    if enable_weekly:
        zones = _add_weekly_confirmation(zones, df, zcfg, logger=logger)
    ...

# inside _add_weekly_confirmation:
weekly_zones = detect(weekly_df, zcfg, logger=None, enable_weekly=False)  # one level only
```

### 6.6 Quarantine look‑ahead in `adjusted_strength`  ·  ML‑integrity
Keep `strength` (formation‑time, leak‑free) as the model feature; rename the future‑aware column to
`adjusted_strength_posthoc` and document that it must **never** be used as a feature at time *t*.
Better: expose a point‑in‑time `test_count_as_of(t)` accessor for backtests.

### 6.7 Reconcile config with philosophy  ·  quality (FP reduction)
Tighten `base_range_multiplier` to ~1.0–1.2 (matches the docstring and shrinks zone width), and raise
`departure_close_ratio` to ~0.5 so it actually filters once §6.2 lowers the body floor.

### 6.8 Make volume optional/auto‑neutral for indices  ·  ranking robustness
When `VolumeRatio` is missing/zero or the instrument is a spot index, set both volume sub‑scores to
neutral and **re‑normalise** the remaining weights so volume noise can't distort the ranking.

### 6.9 Add a trend/HTF quality filter (optional)  ·  FP reduction, aligns with project goals
Record (don't hard‑gate) the zone's alignment with, e.g., price vs `EMA200` / weekly trend, and use it
as a scoring input or an *optional* filter. This addresses the §5.2 counter‑trend false positives the
project instructions care about.

---

## 7. Top 10 highest‑impact changes (with measured coverage estimates)

Baseline = **44 zones**. "Isolated" = change alone; "combined" = stacked with the rows above it.

| # | Change | Type | Isolated coverage | Combined / target | Confidence |
|---|---|---|---:|---:|---|
| 1 | **Detect all 4 patterns (RBR/DBD)** — decouple polarity from arrival (§6.1) | Recall / structural | **44 → 85 (+93%)** | 85 | High (measured) |
| 2 | **Multi‑candle leg‑out + lower per‑candle floor** (§6.2) | Recall / correctness | 44 → 96 (+118%) at 0.6; 56 (+27%) leg‑out | +continuation → ~150–174 | High (measured) |
| 3 | **Arrival = soft / net‑move only** (drop majority gate) (§6.3) | Recall | 44 → 87 (+98%) | with #1+#2 → **214 (≈4.9×)** | High (measured) |
| 4 | **Windowed base search** (fixes greedy + swallow) (§6.4, §3.3/§3.4) | Recall + boundary accuracy | +40 bases now (~0 zones); larger after #2 | all‑in → **~220 (≈5×)** | Med (compounding) |
| 5 | **Weekly recursion guard** (§6.5) | Robustness / perf | 0% (correctness/speed) | — | High (measured: ~44‑deep) |
| 6 | **Quarantine `adjusted_strength` leakage** (§6.6) | ML integrity | 0% (prevents invalid results) | — | High |
| 7 | **Tighten `base_range_multiplier` to ~1.0–1.2** (§6.7) | Precision (−FP) | −zone width; modest −count | net quality ↑ | Med |
| 8 | **Raise `departure_close_ratio` to ~0.5** (currently inert) (§6.6/§3.6) | Precision (−FP) | 0 now → becomes active after #2 | quality ↑ | High (measured inert) |
| 9 | **Volume auto‑neutral + re‑normalise for indices** (§6.8) | Ranking robustness | ranking only | — | Med |
| 10 | **Trend/HTF quality filter** (§6.9) | Precision (−FP) | tunable | aligns with project goals | Med |

**Net expected effect:** items **1–4 together move detection from ~44 to ~150–220 zones (≈3.5–5×, i.e.
~15–20/year)** on Nifty daily — a realistic, quality‑controlled level rather than the current ~4/year.
Items 5–10 don't add raw count; they protect correctness, prevent look‑ahead in the eventual ML stage,
and raise the *precision* of what's detected (fewer wide/counter‑trend/volume‑noise false positives).

**Suggested order of work:** #1 (biggest, cleanest) → #2 → #3 → #4 (compounds #2/#3) → #6 & #5
(integrity/robustness before any ML) → #7/#8/#9/#10 (precision tuning, ideally validated on a
walk‑forward split, not the full history).

---

## 8. What is already correct (so it isn't broken in "fixes")
- **No look‑ahead in detection itself** — formation is pinned to the departure close (lines 902‑904).
- **ATR is Wilder's, backward‑only**; weekly volume MA is backward `rolling`. Good.
- **Weekly confirmation respects causality** (`formation_date(weekly) ≤ formation_date(daily)`,
  lines 548‑551).
- **`_score_zone` uses only formation‑time inputs** (the leak is confined to `adjusted_strength`).
- **Demand/supply mutual exclusivity** is sound (a candle is bullish xor bearish).

Preserve these when refactoring.

---

## 9. Implementation log — fixes #2–#10 applied & verified (2026‑06‑08)

Fix **#1 (continuation patterns) was intentionally NOT applied** at the user's
request; the code below keeps the reversal‑only DBR/RBD model but is written to
compose with #1 later (polarity is already determined by the departure, and the
arrival now returns a direction‑agnostic cleanliness score).

**Files changed:** `src/zones/zone_detector.py`, `config/zone_config.yaml`.

| # | Change | Where | Status |
|---|---|---|---|
| #2 | Multi‑candle leg‑out departure (soft body floor + cumulative displacement) | `_check_departure`, config `departure_*` | ✅ |
| #3 | Soft arrival — net‑move hard gate only; majority folded into score as `arrival_cleanliness` | `_check_arrival`, `_score_zone` | ✅ |
| #4 | End‑anchored windowed base search; removes the `i=j` swallow & grid dependence | `_find_bases` | ✅ |
| #5 | `enable_weekly` guard — weekly recursion now exactly **1 level** (was ~44) | `detect`, `_add_weekly_confirmation` | ✅ |
| #6 | Look‑ahead quarantine: `strength`/`strength_pit` (ML‑safe) vs `adjusted_strength_posthoc` (analysis‑only) | `detect` | ✅ |
| #7 | `base_range_multiplier` 1.8 → **1.2** | config | ✅ |
| #8 | `departure_close_ratio` 0.3 → **0.5** (was inert) | config | ✅ |
| #9 | Volume auto‑neutral + weight renormalisation for untrusted/index volume | `detect`, `_score_zone`, config `volume` | ✅ |
| #10 | Trend alignment (`trend_score`, `trend_aligned`) as a scoring input, never a gate | `_trend_score`, `_score_zone`, config `trend` | ✅ |

**Verified results on ^NSEI (2,812 daily candles):**

- **Coverage: 44 → 97 zones (+120%, 2.2×)**; ~3.85 → **8.5 zones/year**. Status mix
  improved from **0 active** to 6 active / 4 tested / 2 consumed / 85 invalid.
- **#5:** instrumented `detect()` recursion **max depth = 2** (daily + one weekly),
  2 total calls; runtime 0.66 s → **0.24 s**.
- **#6:** `strength` is **provably leak‑free** — recomputed identically on truncated
  data (53/53 zones match, 0 mismatches); `strength_pit ≥ strength`,
  `adjusted_strength_posthoc ≤ strength_pit` hold for all zones.
- **#9:** ^NSEI volume auto‑disabled (1.03 % zero‑volume > 1 % threshold);
  RELIANCE.NS volume retained (0.15 %), volume path runs clean (125 zones).
- **#4:** 0 duplicate departures, 0 identical zones (de‑dup by construction).
- **Threshold enforcement in output:** `departure_leg_atr` min 1.026 (≥ 1.0),
  `departure_close_ratio` min 0.556 (≥ 0.5), `width_atr` max 1.19 (≤ 1.2).
- New CSV schema: 30 columns incl. `departure_leg_atr`, `arrival_cleanliness`,
  `trend_score`, `trend_aligned`, `strength_pit`, `adjusted_strength_posthoc`.

**Note on the column rename (#6):** `adjusted_strength` → `adjusted_strength_posthoc`,
and new `strength_pit`. The only repo reference to the old name was a stale printed
cell in `notebooks/02_zone_detection.ipynb` (output, not executable) — no code break.
Re‑run that notebook to refresh it.

**Tuning knobs if you want more/fewer zones:** lower `departure_leg_disp`
(1.0 → 0.8) or `arrival_min_move` for more; raise `base_range_multiplier` back
toward 1.5 to admit wider bases. Applying #1 on top is projected to roughly double
the count again (~150–220).

---

## 10. Coverage increase — continuation patterns (#1) + data‑driven parameter tuning (2026‑06‑08)

Follow‑up goal: "many visible supply/demand zones are not being formed — increase
the count while keeping accuracy correct."

### 10.1 Algorithm change: #1 implemented (continuation + decoupled arrival)
Polarity is now set by the **leg‑out direction** alone, and the arrival leg is
**direction‑agnostic** with a configurable `arrival_mode`:

- `reversal` — arrival must oppose the leg‑out → only **DBR/RBD** (old behaviour)
- `any` — any directional arrival → **adds RBR/DBD continuation** *(chosen)*
- `optional` — arrival not required → also keeps bare base→leg‑out zones

Each zone now carries a `structure` label (DBR/RBR/RBD/DBD or B*). On ^NSEI the
previously‑undetectable continuation zones are now **96 of 234** (RBR 44 + DBD 52).

### 10.2 Accuracy metric (so "more zones" doesn't mean "worse zones")
With no labelled zones, accuracy is proxied by a **reaction rate**: walk forward
from formation; at the **first retest within 250 bars**, a zone "reacted" if price
made a favorable move ≥ 1×ATR (demand bounces above its top / supply rejects below
its bottom) **before** a close‑based invalidation. `reaction_rate = reacted /
tested`. Validated against a **null model** (random levels, same widths/count):

| Instrument | Detected reaction@1ATR | Null model | Edge |
|---|---:|---:|---:|
| ^NSEI | 0.71–0.75 | ~0.50–0.55 | **≈ +20 pts** |
| RELIANCE.NS | 0.72–0.76 | ~0.48–0.53 | **≈ +22 pts** |

The ~20‑point edge over random confirms the zones carry real predictive structure.

### 10.3 Parameter sweep (72 configs, evaluated on ^NSEI, cross‑checked on RELIANCE)
Swept `arrival_mode × departure_leg_disp × base_range_multiplier × arrival_min_move`.
Representative finalists (NSEI / RELIANCE):

| Config (mode/leg/base) | NSEI count | NSEI react | REL count | REL react | med width(ATR) |
|---|---:|---:|---:|---:|---:|
| reversal / 0.8 / 1.2 (old style) | 145 | 0.756 | 189 | 0.720 | 0.79–0.89 |
| any / 0.8 / 1.2 | 244 | 0.706 | 342 | 0.731 | 0.79–0.86 |
| any / 0.8 / 1.3 | 244 | 0.717 | 350 | 0.728 | 0.84–0.87 |
| **any / 0.8 / 1.5 (CHOSEN)** | **234** | **0.745** | **349** | **0.745** | 0.88–0.91 |
| optional / 0.8 / 1.2 (max count) | 296 | 0.702 | 434 | 0.737 | 0.79–0.86 |

**Selection rule:** maximise count subject to reaction_rate ≥ the reversal‑only
baseline (0.728) on **both** instruments. `any / 0.8 / 1.5` is the only high‑count
config that holds accuracy ≥ baseline on both (0.745/0.745) and is the most
consistent across instruments, so it is the new default.

### 10.4 Final parameters & result
`arrival_mode: any`, `departure_leg_disp: 0.8`, `base_range_multiplier: 1.5`,
`arrival_min_move: 0.3`, `departure_strength: 0.5`, `departure_close_ratio: 0.5`.

| Metric | Original | After #2–#10 | **After #1 + tuning** |
|---|---:|---:|---:|
| Zones (^NSEI, 11y) | 44 | 97 | **234** (5.3× original) |
| Zones/year | 3.9 | 8.5 | **20.9** |
| Continuation (RBR/DBD) | 0 | 0 | **96** |
| Active (usable) zones | 0 | 6 | **13** |
| Reaction rate @1ATR | ~0.73 | ~0.73 | **0.745** (≥ baseline) |

See `data/zones/zone_overlay_comparison.png` for the before/after candlestick
overlay (7 → 22 zones visible in the last 220‑candle window).

**To go further (with the accuracy guardrail):** switch `arrival_mode: optional`
(~296 NSEI / ~434 REL zones at ~0.70 — slightly below baseline accuracy), or lower
`departure_leg_disp` to 0.6. To prioritise R:R over count, set
`base_range_multiplier: 1.2–1.3` (tighter zones, ~same count, ~0.01 lower reaction).
Re‑run the sweep (`/tmp/sweep.py` logic) whenever you change instruments or timeframe.

### 10.5 Reverted to DBR/RBD only + reversal‑specific retune (per user request)

The user only wants the two **reversal** patterns — **DBR** (Drop‑Base‑Rally →
demand) and **RBD** (Rally‑Base‑Drop → supply). Continuation patterns (RBR/DBD)
are disabled via `arrival_mode: reversal` (the leg‑in must oppose the leg‑out), so
the output now contains **only DBR/RBD**. Parameters were then re‑swept *within*
reversal‑only and validated on both instruments against a null model:

| Config (leg/base/min/lb) | NSEI count | NSEI react | NSEI null | REL count | REL react | med width(ATR) |
|---|---:|---:|---:|---:|---:|---:|
| baseline 1.0/1.2/0.5/4 | 97 | 0.728 | 0.52 | 125 | 0.750 | 0.83 |
| **0.6/1.3/0.2/6 (CHOSEN)** | **190** | **0.733** | 0.54 | **262** | 0.680 | **0.83** |
| 0.6/1.5/0.2/6 | 187 | 0.741 | 0.49 | 262 | 0.685 | 0.86 |
| 0.4/1.5/0.3/6 (max count) | 206 | 0.720 | 0.49 | 311 | 0.646 | 0.90 |

**Chosen:** `departure_leg_disp 0.6`, `base_range_multiplier 1.3`,
`arrival_min_move 0.2`, `arrival_lookback 6`, `arrival_mode reversal`
(departure_strength 0.5, departure_close_ratio 0.5 unchanged). Rule: maximise
DBR/RBD count while reaction ≥ baseline on NSEI and zones stay tight.

**Result (^NSEI, DBR/RBD only):**

| Metric | Reversal baseline | **Reversal retuned** |
|---|---:|---:|
| Zones (11y) | 97 | **190** (1.96×) |
| Structure | DBR/RBD | **DBR 87 / RBD 103** (no RBR/DBD) |
| Active zones | 0 | **9** |
| Reaction @1ATR | 0.728 | **0.733** (≥ baseline) |
| Null model | 0.52 | 0.54 (edge ≈ +0.19) |
| Median width (ATR) | 0.83 | **0.83** (unchanged) |

So zone count nearly **doubled** with reaction accuracy held at baseline and zone
tightness unchanged. The before/after chart (`data/zones/zone_overlay_comparison.png`)
shows 7 → 20 DBR/RBD zones in the last‑220‑candle window.

**Note on RELIANCE:** count rose 125 → 262 but reaction eased 0.750 → 0.680 (still
well above the ~0.51 null). If you want RELIANCE accuracy nearer its baseline at the
cost of fewer zones, raise `departure_leg_disp` to 0.8 (≈138 NSEI / ~200 REL at
~0.74–0.79). The `arrival_mode`/`any`/`optional` machinery remains in the code but
is inert in `reversal` mode — only DBR/RBD are emitted.

---

## 11. Weekly data from yfinance + causal weekly‑confluence features (2026‑06‑08)

Two requested changes: (1) fetch the weekly timeframe **directly from Yahoo
Finance** in the data pipeline instead of resampling it inside the detector, and
(2) replace the single `weekly_confirmed` boolean with a richer **causal
weekly‑confluence feature block**.

### 11.1 Weekly data is now fetched, not resampled
`data_pipeline.py` runs a third step that fetches `interval="1wk"` bars from
Yahoo and runs the **same preprocessor** on them, producing
`data/processed/<SYMBOL>_weekly.csv` (alongside the daily file). Threaded through
`fetch_data.py` (`interval`, `suffix`) and `preprocessor.py` (`suffix`,
`min_candles_override`). Config: `data.fetch_weekly`, `data.weekly_interval`,
`preprocessing.min_candles_weekly`. The detector loads this weekly file
(`run_detection`) and passes it to `detect(..., weekly_df=...)`. Resampling
remains only as a fallback when the weekly file is absent.

### 11.2 `weekly_confirmed` → weekly‑confluence feature block
`_add_weekly_confirmation` (single boolean + flat bonus) is replaced by
`_add_weekly_confluence`, which attaches to every daily zone:

| Feature | Meaning |
|---|---|
| `weekly_trend_align` | signed weekly trend distance at formation (>0 = HTF agrees with the zone) |
| `weekly_in_zone` | daily zone overlaps a same‑type weekly zone ≥ `overlap_tolerance` |
| `weekly_dist_atr` | distance (daily‑ATR units) to nearest same‑type weekly zone, 0 if inside |
| `weekly_zone_strength` | strength of the matched weekly zone (inherit HTF quality) |
| `weekly_zone_fresh` | matched weekly zone untested as of the daily formation date |
| `weekly_confluence_score` | 0..1 blend of the above (ranking / `strength_pit`) |
| `weekly_confirmed` | kept = `weekly_in_zone` (backward compatibility) |

`strength_pit` now adds `strength_bonus × weekly_confluence_score` (was a flat
bonus). Config gains `weekly_confirmation.trend_ma`.

### 11.3 Causality (no look‑ahead) — verified
Every feature uses only weekly bars/zones knowable by the daily zone's formation
date (weekly zones filtered to `formation_date ≤ T`; trend from the last weekly
bar ≤ T; freshness counts weekly touches only in `(weekly_formation, T]`).
**Verified empirically:** re‑running with all future daily *and* weekly data
truncated reproduced `strength_pit` and `weekly_confluence_score` **exactly for
all 89 early zones (0 mismatches)**. So `strength`/`strength_pit` remain ML‑safe.

### 11.4 Result & signal
On ^NSEI (190 DBR/RBD zones): 37 sit inside a weekly zone, 22 of those are fresh;
mean confluence 0.331. Zones with **above‑median confluence reacted 0.750 vs
0.718** below median — a modest but positive edge, consistent with weekly
confluence being a secondary quality signal (best used as an ML feature, which is
exactly how it is now exposed). See `data/zones/weekly_confluence_overlay.png`.

> **Note on the weekly file in this sandbox.** Yahoo Finance is unreachable from
> this environment, so the weekly file used for testing was bootstrapped by
> resampling the daily data through the real preprocessor (a close stand‑in). On
> your machine, run `python data_pipeline.py --symbol ^NSEI` once to pull the
> **real** 1wk bars (it will overwrite the bootstrap), then re‑run detection.

---

## 12. Base‑location fix — bases truncated to one candle (2026‑06‑08)

User feedback: on the charts the marked base was often **1–2 candles too late** —
the true consolidation sat a couple of candles behind where the algorithm drew it.

**Root cause (confirmed on data).** The aggregate "cluster" gate reused the *same*
`base_range_multiplier` (1.3 ATR) as the per‑candle gate. A genuine multi‑candle
consolidation of individually‑tight candles legitimately spans ~1.5–2.2 ATR, so the
windowed search kept shrinking the base until only the single candle next to the
departure fit under 1.3 ATR. Measured: **61% of bases were single‑candle**, and
**53 zones were truncated specifically by this gate**. Example — supply zone at idx
67: candles 63–67 form a clean 4–5 candle top, but only candle 67 was tagged as the
base.

**Fix.** Separate the two gates and allow longer bases:
- new `base_cluster_multiplier: 2.5` — aggregate span budget for the whole base
  (per‑candle `base_range_multiplier` stays 1.3);
- `max_base_length: 3 → 5`;
- base‑tightness score now scales by the cluster multiplier.

`_find_bases` now uses the cluster gate for the windowed aggregate check; the base
extends backward over the full consolidation.

**Result (^NSEI):**

| Metric | Before fix | After fix |
|---|---:|---:|
| Single‑candle bases | 61% | **48%** |
| base_length spread | 1:115 2:42 3:33 | **1:85 2:40 3:20 4:12 5:19** |
| Zones | 190 | 176 |
| Reaction @1ATR | 0.733 | **0.735** (held) |
| Median width (ATR) | 0.83 | 0.94 |

Example (idx 67) now captures `[63..67]` (length 5) instead of `[67]`. The 14
dropped zones had departures that no longer cleared the *fuller* base by
`departure_leg_disp` — i.e. weak leg‑outs that should not have qualified. Accuracy
held, so the bases are more correct without costing reaction quality. See
`data/zones/base_location_fix.png`. Tune `base_cluster_multiplier` down (→2.0) for
tighter zones or up (→3.0) to capture even wider consolidations.
