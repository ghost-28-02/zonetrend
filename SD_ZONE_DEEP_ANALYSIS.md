# Supply/Demand Zone Detection — Deep Algorithmic Analysis

**Scope:** the live code in `src/zones/zone_detector.py` + `config/zone_config.yaml`, evaluated on
the real dataset `data/processed/IDX_NSEI.csv` (2,812 daily candles) with weekly confluence from
`IDX_NSEI_weekly.csv`. Strictly **DBR/RBD** (reversal) mode.

Every quantitative claim below comes from running the *actual* detector + an out‑of‑sample
**reaction‑rate** quality metric, not from theory.

> **Reaction‑rate metric (the quality yardstick used throughout).** For each zone, walk forward to
> the first retest within 250 bars; the zone "reacted" if price moved ≥ 1×ATR in the expected
> direction (demand bounces above its top / supply rejects below its bottom) **before** a
> close‑based invalidation. `reaction_rate = reacted / (reacted+failed+weak)`. A null model of
> random levels scores ~0.50, so the detector's ~0.73 carries a real ~+20pt edge. This is a *proxy*
> for "institutional zone", not ground truth — but it is consistent and lets us rank changes.

---

## 0. The headline finding (read this first)

**The current `strength` score is anti‑predictive.** Sorted by `strength`:

| Bucket | Reaction rate |
|---|---:|
| Top‑quartile strength | **0.697** |
| Bottom‑quartile strength | **0.872** |

High‑scoring zones react *worse* than low‑scoring ones. Confirmed independently: zones that
**failed** carry a *higher* mean strength (0.759) than zones that **reacted** (0.734). The cause is
structural — the score is dominated by **departure magnitude** (weight 0.40, half of it the leg‑out
displacement), and on this data **bigger displacement predicts worse reaction** (leg 0.8–1.2 ATR →
0.796; leg > 2 ATR → 0.652). The detector is, in effect, ranking "price already ran a long way"
highest, which is exactly the lowest‑probability retest.

Everything in §7 (new scoring) flows from this. Two more results overturn stated assumptions:

- The fixed **3‑candle leg‑out cap drops ~95 otherwise‑valid zones** whose impulse simply took 4–8
  candles to clear the base (§4, §5). This is the largest single false‑negative source.
- The proposed **base‑containment rule** (`highest_base_high < close of both legs`), taken
  literally, selects *worse* zones (contained 0.684 vs violators 0.742). The intent is right; the
  formulation is not (§3.3).

---

## 1. Current architecture & exact filter logic

Pipeline in `detect()`: `_find_bases` → for each base try `_check_departure` (sets polarity) →
`_check_arrival` (reversal gate) → `_score_zone` → `_track_zone_status` → `_add_weekly_confluence`
→ strict DBR/RBD guard.

| Stage | Function | Exact rule (current values) |
|---|---|---|
| Base candle | `_is_base_candle` | `(H−L) ≤ 1.3·ATR` **and** `|C−O| ≤ 0.5·ATR` |
| Base group | `_find_bases` | end‑anchored window; per‑candle gate 1.3, **cluster** gate `(maxH−minL) ≤ 2.5·ATR`, length 1–5 |
| Departure | `_check_departure` | first post‑base candle in‑direction, close‑ratio ≥ 0.5, body ≥ 0.5·ATR; **then** close displaced ≥ 0.6·ATR beyond base edge **within 3 candles** |
| Arrival | `_check_arrival` | net close move over **fixed 6 candles** ≥ 0.2·ATR, opposite to leg‑out (reversal) |
| Score | `_score_zone` | 0.40 departure + 0.25 base‑tightness + 0.20 arrival + 0.15 volume + 0.10 trend (renormalised) |
| Boundaries | `detect` | **full wick range**: `top=max(High)`, `bottom=min(Low)` of base candles |

Measured funnel: **1,694 bases → 326 departure‑passes → 176 final zones** (88 DBR / 88 RBD),
overall reaction **0.735** (147 retested of 176).

---

## 2. Current weaknesses (Output 1)

1. **Inverted quality score** (§0). The single biggest correctness problem — the score should be the
   product, not an after‑thought.
2. **Fixed‑window legs.** Both legs are hard windows (leg‑out ≤ 3, leg‑in = 6). Real impulses are
   1–N candles; 25% of leg‑outs structurally run longer than 3 (§4). No swing/structure termination.
3. **Single‑candle bases dominate and underperform.** 48% of bases are 1 candle; they react 0.682
   vs 0.763–0.875 for 2–5 candle bases (§3.1). The detector over‑produces the weakest base type.
4. **No per‑leg metrics.** None of `leg_in/out_distance`, `leg_in/out_candles`, velocity, momentum,
   displacement/base ratio are computed or stored — so neither scoring nor research can use them.
5. **No zone merging.** 18% of zones (32) are ≥50%-overlapping same‑type duplicates within 60 bars
   (§9). They inflate counts and clutter charts.
6. **Boundaries use the full wick range**, which on a wicky base over‑widens the zone (worsens R:R)
   and is not the distal/proximal convention practitioners use (§6).
7. **Volume is inert on the primary instrument** (index volume untrusted → dropped), so 25% of the
   nominal score weight silently collapses to the other terms.
8. **No V‑shape / reversal‑strength concept, no swing‑structure validation.**

---

## 3. Filter‑by‑filter evaluation (too strict / too loose)

### 3.1 Base detection — *mostly right after the recent fix, but length is under‑used*
Per‑candle (1.3 range / 0.5 body) and the separate cluster gate (2.5) are reasonable; the cluster
fix already lifted multi‑candle bases (single‑candle 61%→48%). **The remaining problem is that
single‑candle bases are still emitted with equal standing despite reacting worst:**

| base_length | n | reaction |
|---:|---:|---:|
| 1 | 85 | **0.682** |
| 2 | 40 | 0.763 |
| 3 | 20 | 0.706 |
| 4 | 12 | **0.875** |
| 5 | 19 | 0.833 |

**Verdict:** not "too strict/loose" on width, but **length is a strong quality axis that is ignored**.
Recommend: keep 1‑candle bases (some are sharp imbalances) but **penalise them in scoring**, and
require a *stronger* departure for length‑1 (see §7).

Base **wick‑fraction** is also predictive and unused: body‑heavy bases react 0.667, wicky bases
(wick fraction > 0.7) react **0.797**. Long base wicks = order absorption → add to score.

### 3.2 Departure / leg‑out — **`leg_max=3` is the most damaging "too strict" filter**
The displacement floor (0.6 ATR) is fine — but the **3‑candle window is not**. Of bases whose first
candle already qualifies (direction/close/body), **~95 are rejected only because the close reached
0.6·ATR beyond the base after candle 3 (but within 8).** That is the dominant false‑negative source.
And displacement is **non‑monotonic** — requiring *more* would hurt:

| leg‑out displacement | n | reaction |
|---|---:|---:|
| < 0.8 ATR | 31 | 0.741 |
| 0.8–1.2 ATR | 60 | **0.796** |
| 1.2–2 ATR | 52 | 0.698 |
| > 2 ATR | 33 | **0.652** |

**Verdict:** window **too strict** (fix with dynamic legs, §4); magnitude floor about right — do
**not** raise it; instead **reward the 0.8–1.5 ATR sweet spot** and stop rewarding > 2 ATR.

### 3.3 The proposed base‑containment rule — **tested; not optimal as stated**
Rule (DBR): `highest_base_high < close_prev_bearish_leg AND < close_next_bullish_leg`. Implemented
literally (base high below the candle before the base *and* below the departure close):

| group | n | reaction |
|---|---:|---:|
| "contained" (passes rule) | 20 | **0.684** |
| violators | 156 | 0.742 |

The rule **rejects the better zones**: only 20/176 pass, and they react *worse*. Reason: for a
demand base, `base_high < close[base_start−1]` only holds when the base gaps strictly below the prior
candle — an unusual, gappy structure that is *not* higher quality here. The **intent** (base sits
inside the imbalance / is consumed by the leg‑out) is sound and is **already enforced better** by the
displacement test (`close clears base edge by ≥ leg_disp`). **Recommendation:** drop the literal
containment inequality; instead keep/strengthen "**leg‑out close clears the base proximal line by
≥ leg_disp**" and add "**leg‑in originates outside the base**" (close of the candle before the base is
on the far side of the base). That captures the same idea without the gappy bias.

### 3.4 Arrival — *too rigid, low value as a hard gate*
Fixed 6‑candle net‑move ≥ 0.2 ATR. The 0.2 floor is loose (rarely binds); the **fixed window** is the
issue (same dynamic‑leg problem). Because polarity comes from the departure, the arrival's real job
is reversal/continuation classification and *context scoring*, not gating. Recommend making leg‑in
**dynamic** (§4) and keeping it a soft/score input except for the reversal‑direction requirement.

---

## 4. Dynamic Leg‑In / Leg‑Out (the core architectural fix)

**Problem:** both legs are fixed windows, so a 4–8 candle institutional impulse is truncated
(false negatives) and `leg_*_candles/distance` are never the *true* impulse. Measured leg‑out length
(structural): 1→55, 2→44, 3→33, 4→21, 5→9, 6→7, 7→6, 9→1 — **25% exceed the cap**.

**Robust implementation — extend until structure changes, then measure the full impulse.** Use a
swing/pullback rule (no look‑ahead beyond the leg itself, which is fine — the zone only *forms* once
the leg has displaced past the base, and you can cap the search horizon):

```python
def _trace_leg(df, start_idx, direction, atr, max_len=12, pullback_atr=0.6):
    """Extend a leg from start_idx in `direction` ('up'/'down') until the impulse ends:
    a counter-move (close) of >= pullback_atr*ATR from the running extreme, or a lower-high/
    higher-low swing. Returns (end_idx, candles, displacement_atr, body_sum, vol_sum)."""
    C = df['Close'].values; H=df['High'].values; L=df['Low'].values; V=df['Volume'].values
    n=len(df); ext = C[start_idx]; ext_idx=start_idx; body=0.0; vol=0.0
    for i in range(start_idx, min(n, start_idx+max_len)):
        body += abs(C[i]-df['Open'].values[i]); vol += V[i]
        if direction=='up':
            if C[i] > ext: ext, ext_idx = C[i], i
            elif (ext - C[i]) >= pullback_atr*atr: break          # impulse exhausted
        else:
            if C[i] < ext: ext, ext_idx = C[i], i
            elif (C[i] - ext) >= pullback_atr*atr: break
    candles = ext_idx - start_idx + 1
    disp = abs(ext - C[start_idx]) / atr
    return ext_idx, candles, round(disp,4), body, vol
```

Wire it in: leg‑out = `_trace_leg(df, base_end+1, leg_dir, atr)`; leg‑in = trace **backward** from
`base_start−1`. Then the departure test becomes "leg‑out **displacement** (not a 3‑candle max) clears
the base proximal edge by ≥ leg_disp", and the metrics in §10 fall out for free.

**Expected impact:** recovers a large share of the ~95 leg‑length false negatives → **+30–50% zone
count**; gives true `leg_*_candles/distance` for scoring; lets V‑shape (§5) and velocity be computed.
Guardrail: keep the moderate‑displacement scoring (§7) so the extra long‑leg zones are *ranked*, not
blindly trusted.

---

## 5. Per‑leg metrics & displacement method (Outputs for "Metrics" + "Displacement")

Compute and **store** per leg (cheap once `_trace_leg` exists): `leg_in/out_distance` (price &
ATR‑normalised), `leg_in/out_candles`, `pct_move = dist/close`, `avg_body = body_sum/candles`,
`velocity = displacement_atr / candles`, `volume_expansion = leg_vol_avg / base_vol_avg`,
`momentum = Σ signed body / Σ range`, and `displacement_base_ratio = leg_out_distance / base_height`.

**Which displacement method wins (measured):** ATR‑multiple and displacement/base ratio are **both
non‑monotonic** here, and "bigger is better" is false:

| displacement/base | n | reaction | | ATR multiple | n | reaction |
|---|---:|---:|---|---|---:|---:|
| < 1.0 | 56 | **0.804** | | < 0.8 | 31 | 0.741 |
| 1–2 | 78 | 0.731 | | 0.8–1.2 | 60 | **0.796** |
| 2–4 | 33 | **0.577** | | 1.2–2 | 52 | 0.698 |
| > 4 | 9 | 1.000* | | > 2 | 33 | **0.652** |

*small sample. **Recommendation:** use **ATR‑multiple as the gate** (robust, volatility‑normalised,
sweet spot 0.8–1.5) and **displacement/base ratio + velocity as score inputs with a peaked
(not increasing) response** — reward moderate, penalise extreme. Swing‑to‑swing distance (from
`_trace_leg`) is the *correct denominator* for these ratios and replaces today's 3‑candle proxy.
Percentage move adds nothing beyond ATR‑normalisation on a single instrument; keep it only for
cross‑instrument comparability.

---

## 6. Zone boundaries (Output: boundary method)

Current = full wick range (max High / min Low of base). Evaluated against the four options:

| Method | Pro | Con / data note |
|---|---|---|
| Full wick | never "misses" a touch | over‑wide on wicky bases → worse R:R; today's median width 0.94 ATR |
| Body range | tight entries | misses wick‑based reactions (price often turns at the wick) |
| Distal/Proximal | the practitioner standard | needs a clear proximal edge definition |
| **Hybrid (recommend)** | tight but safe | **proximal = body edge nearest the leg‑out; distal = furthest wick** |

**Recommendation — hybrid:** *proximal line* (entry trigger) = the base **body** edge on the
departure side; *distal line* (stop) = the **furthest wick**. This narrows the actionable zone
(better R:R) while keeping the stop behind the wick. It also makes the §7 wick‑fraction signal
actionable. Store both `proximal`/`distal` plus the current `top/bottom`.

---

## 7. New scoring system, 0–100 (Output 5/6) — rebuilt from the data

The current weights are anti‑predictive (§0). Rebuild around what actually predicts reaction here:
**base length (↑), base wickiness (↑), moderate leg‑out displacement (peaked), freshness (↑),
HTF confluence (↑, modest), and a penalty for over‑extension.** Proposed:

```
quality(0..100) = 100 * Σ wᵢ·scoreᵢ ,  Σwᵢ = 1
```

| Component | Weight | score definition (0..1) | why (data) |
|---|---:|---|---|
| Base structure | **0.22** | 0.6·len_score + 0.4·wick_frac; len_score: 1→0.4, 2→0.7, 3→0.8, 4–5→1.0 | len & wick both strongly predictive (§3.1) |
| Leg‑out quality | **0.20** | peaked at 1.0 for disp 0.8–1.5 ATR, decaying to ~0.4 by 0.4 and by 2.5 ATR | moderate disp reacts best (§5) |
| Leg‑in quality | 0.10 | velocity & cleanliness of the (dynamic) leg‑in | momentum into the base |
| Displacement/base | 0.08 | peaked near 1–1.5, penalise > 3 | non‑monotonic (§5) |
| Freshness | **0.15** | 1.0 fresh → 0.85 → 0.65 → 0.45 (by retest count, point‑in‑time) | exhaustion is real |
| HTF confluence | 0.10 | existing `weekly_confluence_score` | secondary, treat cautiously* |
| Volume expansion | 0.07 | leg‑out vol / base vol (skip→neutral on indices) | only when trusted |
| V‑shape bonus | 0.08 | symmetry × reversal strength (§8) | sharp reversals = stronger imbalance |

*On this index, weekly confluence was *slightly* inverted (fail 0.40 vs react 0.34) on a small
sample — keep the weight modest and re‑measure on equities before trusting it. **Validation of the
direction:** a simple high‑quality slice `base_len ≥ 2 AND leg_out 0.7–1.6 ATR` already lifts
reaction **0.735 → 0.833 (n=55)** — i.e. re‑weighting toward these axes demonstrably concentrates
quality. Normalise to 0–100 and expose `quality_score`; keep raw `strength` for backward compat but
**stop using it for ranking**.

---

## 8. V‑shape detection (Output)

Define a V‑zone as a sharp reversal: leg‑in and leg‑out **both** ≥ `v_disp` (e.g. 1.5 ATR) within a
short span and roughly symmetric. Using the dynamic legs:

```python
v_in  = leg_in.displacement_atr;  v_out = leg_out.displacement_atr
symmetry = min(v_in, v_out) / max(v_in, v_out)            # 1.0 = perfectly symmetric
is_v = (v_in >= 1.5 and v_out >= 1.5 and symmetry >= 0.6
        and leg_in.candles + leg_out.candles <= 10)
v_strength = symmetry * min(v_in, v_out) / 1.5            # 0..~1
```

**Should V‑zones score higher?** Yes, but as a *bounded bonus* (weight 0.08), not an override —
because §5 shows very large displacement legs react worse, so an *unbounded* V reward would
re‑introduce the inversion. Reward **symmetry and clean reversal**, cap the displacement contribution.

---

## 9. Zone merging (Output) — implement it (none exists)

18% of zones (32, in 26 clusters) are ≥ 50% overlapping same‑type within 60 bars. Merge:

```python
def merge_zones(zones, overlap=0.5, time_gap=60):
    out=[]; 
    for z in zones.sort_values('formation_idx').itertuples():
        hit=None
        for m in out:
            if m['type']!=z.type: continue
            inter=min(m['top'],z.top)-max(m['bottom'],z.bottom)
            h=min(m['top']-m['bottom'], z.top-z.bottom)
            if h>0 and inter/h>=overlap and abs(z.formation_idx-m['formation_idx'])<=time_gap:
                hit=m; break
        if hit:   # keep the higher-quality one; widen to the union, keep earliest formation
            if z.quality_score>hit['quality']:
                hit.update(top=z.top,bottom=z.bottom,quality=z.quality_score)
            hit['merged']=hit.get('merged',1)+1
        else:
            out.append(dict(type=z.type,top=z.top,bottom=z.bottom,
                            formation_idx=z.formation_idx,quality=z.quality_score,merged=1))
    return out
```

Thresholds: **overlap ≥ 0.5, time_gap ≤ 60 bars** (tunable). Keep the highest‑quality zone, record
`merged_count` (a confluence signal). Expected: −15–18% raw count, cleaner non‑redundant set.

---

## 10. Freshness, volume, swing structure, multi‑timeframe (Outputs)

- **Freshness:** the decay (`[1.0,0.85,0.65,0.45]`) and status tracking are sound and *causal*. Add
  the explicit labels the spec asks for: `fresh (0 tests) / tested_once / tested_twice / expired
  (≥ max_test_count or invalidated)`. Keep invalidation = close beyond the **distal** line (not the
  proximal) to avoid premature kills.
- **Volume:** keep the auto‑trust gate (indices → drop). For equities add **leg‑out volume
  expansion** (`leg_out_avg_vol / 20d_MA`) and a **spike** flag (any leg‑out candle > 2× MA) — both
  computable from `_trace_leg`. Weight modestly (0.07) and only when trusted.
- **Swing structure:** use pivots/fractals to *terminate* the dynamic legs (a confirmed swing is the
  cleanest leg boundary) and to *validate* that the base sits at a swing extreme. Drawback: lag
  (a fractal needs k bars to confirm) — acceptable because zones only form after the leg‑out anyway.
- **Multi‑timeframe:** the current **confluence‑as‑features** design is the right default (don't
  hard‑filter — it discards tradeable LTF zones, and HTF confluence was only weakly predictive here).
  Detect each timeframe **independently**, attach weekly features causally (already done), and offer
  an *optional* "HTF‑only" filter for conservative users. Do **not** merge across timeframes into one
  zone — keep them separate and let confluence be a score input.

---

## 11. Recommended parameters, complexity, architecture (Outputs 4, 7, 8)

**Recommended parameter values:**

| Param | Now | Recommend | Reason |
|---|---:|---:|---|
| `departure_leg_max` | 3 | **dynamic (cap 12)** | ~95 false negatives from the 3‑cap (§4) |
| `departure_leg_disp` | 0.6 | 0.6 (gate) + peaked score | magnitude floor fine; don't raise (§5) |
| `arrival_lookback` | 6 | **dynamic** | fixed window truncates leg‑in (§4) |
| `base_*` (1.3/2.5/len 5) | — | keep | post‑fix values are good (§3.1) |
| scoring weights | dep‑heavy | **§7 table** | current weights inverted (§0) |
| `merge.overlap` | — | 0.5 / 60 bars | 18% duplicates (§9) |
| boundaries | full wick | **hybrid distal/proximal** | R:R + actionable (§6) |

**Complexity / performance.** `_find_bases` is `O(n·max_len)`; the detect loop is `O(bases · (leg +
lookback))`; status tracking is `O(zones · horizon)`. For n≈2.8k this runs in ~0.2s — fine. The
hotspots are the **per‑row `df.iloc[i]` calls** inside the departure/arrival/status loops (pandas
scalar access is ~100× slower than numpy). **Optimisation:** pull `.values` arrays once (already done
in `_find_bases`; do the same in `_check_departure/_check_arrival/_track_zone_status`) → ~5–10× on
those loops. Memory is trivial. For backtesting many symbols, vectorise the base scan with rolling
`max/min` and precompute ATR‑normalised arrays once per symbol.

**Architecture (recommend) — split the monolith into testable modules:**
```
zones/
  legs.py        # _trace_leg, leg metrics (dynamic, swing-terminated)
  bases.py       # base detection (per-candle + cluster + length/wick stats)
  patterns.py    # DBR/RBD assembly, V-shape, containment/origin checks
  boundaries.py  # wick/body/distal-proximal/hybrid
  scoring.py     # 0–100 quality model (§7), pure & unit-tested
  freshness.py   # status, decay, labels, invalidation
  htf.py         # weekly confluence (causal)
  merge.py       # overlap merging
  detect.py      # orchestrator
```
This makes each filter independently unit‑testable (today the leakage/causality guarantees live in
one 1.3k‑line file) and lets the scoring model be swapped/back‑tested in isolation.

---

## 12. Concrete changes ranked by expected impact (Outputs 11–12)

| # | Change | Type | Expected impact (measured/estimated) |
|---|---|---|---|
| 1 | **Rebuild scoring (§7)** — base length+wick, peaked displacement, freshness, drop dep‑magnitude dominance | Quality | turns an **inverted** ranker into a real one; quality slice 0.735→**0.833** |
| 2 | **Dynamic legs (§4)** replace 3‑/6‑candle windows | Recall | recover ~95 FN → **+30–50% zones**; true leg metrics |
| 3 | **Penalise single‑candle bases / reward length+wick** (part of #1) | Precision | base_len1 0.682 → length‑weighted set ≥ 0.80 |
| 4 | **Don't raise displacement; reward 0.8–1.5 sweet spot, penalise >2** | Precision | removes the >2 ATR drag (0.652) |
| 5 | **Hybrid distal/proximal boundaries (§6)** | Quality/R:R | tighter entries, stop behind wick |
| 6 | **Zone merging (§9)** | Noise | −18% duplicate zones |
| 7 | **Drop literal base‑containment; use leg‑clears‑proximal + leg‑in‑origin (§3.3)** | Precision | avoids the 0.684 trap |
| 8 | **Per‑leg metrics stored (§5,§10)** | Enables ML | feeds the model & research |
| 9 | **V‑shape bounded bonus (§8)** | Quality | rewards clean reversals without re‑inverting |
| 10 | **Vectorise inner loops; modularise (§11)** | Perf/maintainability | ~5–10× on hot loops; testable filters |

**Suggested order:** #1+#4 (scoring — biggest correctness win, low effort) → #2 (dynamic legs —
biggest recall win) → #5/#6/#7 (precision/cleanliness) → #8/#9 → #10. Re‑run the reaction‑rate
evaluator after each, and **cross‑validate on RELIANCE** (and ≥1 more equity) before locking — the
HTF and volume signals especially need a second instrument, since they were weak/ambiguous on the
index here.

---

## 13. False‑negative & false‑positive summaries (Outputs 2, 3)

**False negatives (valid zones missed):**
- ~95 zones whose leg‑out cleared the base only after candle 3 — *the* dominant cause (§4). Rejection
  by `departure_leg_max`; **incorrect** rejection; fix = dynamic legs.
- Leg‑in truncated by the fixed 6‑candle window (slow approaches). Rejection by `arrival_lookback`.
- (Continuation RBR/DBD are intentionally excluded by strict DBR/RBD mode — *correct* per your spec.)

**False positives (weak zones accepted):**
- **Single‑candle bases** — 48% of output, react 0.682; over‑represented. Filter: length/wick in
  scoring (§7) + minimum quality threshold.
- **Over‑extended legs** (> 2 ATR displacement, react 0.652) scored *highest* by the current model.
  Filter: peaked displacement score (§7).
- **13 zones invalidated before any retest** (price blew straight through) — formed on exhaustion.
  Filter: V‑shape/origin checks + quality threshold.
- **18% overlapping duplicates** (§9). Filter: merging.

The through‑line: the detector's *recall* problem is the fixed leg window, and its *precision*
problem is a scoring model that rewards the wrong axis. Fixing those two (#1, #2) addresses the
majority of both missed institutional zones and low‑quality detections.

---

## 14. Implementation log & empirical corrections (2026‑06‑09)

All changes were implemented **and tested against the reaction metric**. Testing **overturned two of
my own §‑level recommendations** — recorded honestly here.

**Landed and validated (^NSEI):**

| Change | Result |
|---|---|
| **Rebuilt quality score (§7)** → `quality_score` 0–100 | **Inversion fixed**: top‑quartile reaction **0.816** vs bottom‑quartile **0.625** (was inverted 0.697/0.872). The headline win. |
| **Dynamic leg tracing (§4)** `_trace_leg_out/_trace_leg_in` | True per‑leg metrics (`leg_*_candles/disp/velocity`) now stored & used. |
| **Per‑leg metrics (§5/§10)** | `leg_out_disp_atr`, `leg_out_velocity`, `leg_in_*`, `disp_base_ratio` columns added. |
| **Hybrid distal/proximal boundaries (§6)** | `proximal`/`distal` columns added (entry near body edge, stop behind wick). |
| **Zone merging (§9)** | 224 → 161 (63 duplicates merged); `merged_count` recorded. |
| **V‑shape bonus (§8)** | 11 V‑zones flagged (`is_v_shape`, `v_strength`); bounded bonus in score. |
| **Volume / base / wick / trend in scoring** | wickiness + base length now drive quality (the predictive axes). |
| **Leak‑free preserved** | `quality_score` & `strength_pit` identical under data truncation (71/71). Strict DBR/RBD preserved (87 RBD / 74 DBR). |

**Net:** **161 zones @ 0.746 reaction** (vs the previous 176 @ 0.735 — *more accurate*), and the
score now ranks correctly so a `quality_score ≥ 60` tier gives **111 zones @ 0.790**.

**Overturned by testing (reverted) — the important honesty:**

1. **Extending the leg cap to 12 (§4) was WRONG.** Tested: 1–2 candle stalls react **0.53** and 7+
   candle grinds **0.63**, while 3–4 candle legs react **0.73**. The old `leg_max=3` was implicitly
   selecting the sweet spot. Re‑tuned to **`leg_max=4`, `pullback=1.0`** (dynamic mechanism kept for
   metrics; cap kept tight). So the "~95 missed zones" from §4 were mostly *low‑quality* slow legs —
   missing them was fine, not a false‑negative problem.
2. **Forming the zone at the "displacement‑confirmation" candle (§4) HURT** (dropped reaction to
   ~0.61 by shifting the retest window). Reverted to `formation_idx = base_end+1` (the validated
   behaviour). +0.13 reaction recovered.
3. **The leg‑in origin check as a HARD gate (§3.3) lowered reaction** and is **OFF by default**
   (`arrival_require_origin: false`) — consistent with the earlier finding that the literal
   containment rule selected worse zones. `origin_ok` is kept as a recorded feature, not a gate.

**Takeaway:** the previous *detection set* was already near a good frontier; the deep‑analysis's real,
validated payoff was **fixing the quality SCORE** (ranking/precision) plus the additive features
(metrics, boundaries, merging, V‑shape), not the detection‑changing ideas — three of which the data
rejected. Final params: `departure_leg_max: 4`, `leg_pullback_atr: 1.0`, `arrival_require_origin:
false`, new `quality_weights` / `v_shape` / `merge` blocks. Re‑run the sweep on a second instrument
before locking the scoring weights.

---

## 15. V / inverted‑V shape + base containment (user rule, 2026‑06‑09)

Per request, every zone is now labelled by shape — **`shape = "V"` for DBR (demand)** and
**`"inverted_V"` for RBD (supply)** — and a base‑containment gate (`require_v_containment: true`,
configurable) enforces the clean V/inverted‑V geometry:

- **DBR:** base **HIGH** < `open[prev candle]` **and** < `close[first leg‑out candle]`
- **RBD:** base **LOW** > `open[prev candle]` **and** > `close[first leg‑out candle]`

Verified: 0 kept zones violate the rule; labels match type exactly (43 V / 54 inverted‑V).

**Measured impact (honest).** The gate is **strict** and, on this index, *reduces* coverage and the
reaction proxy:

| `require_v_containment` | zones | reaction | quality top‑q / bottom‑q |
|---|---:|---:|---:|
| false | 161 | **0.746** | 0.816 / 0.625 |
| **true (default, per request)** | **97** | 0.620 | 0.714 / 0.500 |

Why it cuts: the second clause forces the **first** leg‑out candle to already close beyond the base,
which rejects the many valid zones whose (dynamic) leg‑out clears the base over 2–4 candles; the first
clause (`base beyond prior candle's open`) rejects bases that overlap the prior candle. Both are
legitimate "clean‑V" constraints — they trade recall/accuracy for geometric purity. It is **on by
default as requested**; set `require_v_containment: false` to recover the 161 @ 0.746 set (the `shape`
labels remain either way). If you want the V geometry without the full accuracy cost, the natural
softening is to test the `close` clause against the leg‑out's *confirming* close rather than strictly
the first candle — happy to wire that as a third mode.

**Update (removed by request).** The V / inverted‑V logic was subsequently removed entirely — the
`require_v_containment` gate, the `shape` column, **and** the older `is_v_shape`/`v_strength` scoring
bonus and `v_shape` config are all gone. Base params were also relaxed
(`base_range_multiplier 1.5`, `base_body_multiplier 0.6`, `base_cluster_multiplier 3.0`). With the
strict gate removed, detection recovered and *improved*: **173 zones @ 0.796 reaction** (best so far),
quality still ranks correctly (top‑quartile 0.833 vs bottom‑quartile 0.722), and `corr(strength,
quality_score) ≈ 0.09` (legacy score remains orthogonal). The detector is back to: dynamic legs +
rebuilt quality score + distal/proximal boundaries + merging + weekly confluence, strictly DBR/RBD.
