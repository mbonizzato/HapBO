# HapBO Prior Specification Language (PSL) — Vocabulary Card

**This document is the *vocabulary*, not the task.** It defines every token you are
allowed to use when you write a **belief document**: a JSON object describing where an
expert thinks the optimum of a search space lies, and how much to trust that belief. A
separate, deterministic compiler (no LLM) turns your belief document into the numbers the
optimizer needs. Your job is only to write a *valid, faithful* belief document in the
language below. You never compute the prior yourself.

Read the whole card once before writing. The grammar is **closed**: if a belief cannot be
said with the tokens here, encode it with `anchors` (§4.12) — never invent a token.

---

## 0. What the document must contain

A belief document is a single JSON object with exactly three top-level keys:

```json
{
  "schema":  { ... },      // the typed search space (what the knobs are)
  "beliefs": [ ... ],      // a list of belief tokens (where the optimum is)
  "trust":   { ... }       // overall confidence metadata
}
```

- **`schema`** names and types every dimension. Beliefs refer to dimensions **by name**
  and give coordinates in the dimension's **own native units** (e.g. `0.001` for a
  continuous knob, `"level_a"` for a categorical knob), never as percentages or grid
  indices. The compiler owns all coordinate geometry; you only speak the expert's nouns.
- **`beliefs`** is an ordered list of tokens from §4. Each non-meta token carries a
  `weight` and a `provenance` (§3). Order does not matter to the compiler.
- **`trust`** gives one overall confidence word and optional notes (§5).

---

## 1. The machine contract (JSON Schema)

You should be decoded against this schema. It is the source of truth for *syntax*; the
prose in §4 is the source of truth for *meaning*. Where the two ever seem to differ, the
schema wins on shape and §4 wins on intent.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "HapBO PSL belief document",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema", "beliefs", "trust"],
  "properties": {
    "schema": {
      "type": "object",
      "additionalProperties": false,
      "required": ["objective", "dimensions"],
      "properties": {
        "objective": { "enum": ["maximize", "minimize"] },
        "dimensions": {
          "type": "array", "minItems": 1,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["name", "type"],
            "properties": {
              "name":    { "type": "string", "pattern": "^[a-z][a-z0-9_]*$" },
              "type":    { "enum": ["continuous", "ordinal", "categorical"] },
              "range":   { "type": "array", "items": { "type": "number" },
                           "minItems": 2, "maxItems": 2 },
              "values":  { "type": "array", "minItems": 1 },
              "scale":   { "enum": ["linear", "log"] },
              "meaning": { "type": "string" }
            },
            "allOf": [
              { "if":   { "properties": { "type": { "const": "continuous" } } },
                "then": { "required": ["range"] } },
              { "if":   { "properties": { "type": { "enum": ["ordinal", "categorical"] } } },
                "then": { "required": ["values"] } }
            ]
          }
        }
      }
    },

    "beliefs": {
      "type": "array",
      "items": { "$ref": "#/$defs/belief" }
    },

    "trust": {
      "type": "object",
      "additionalProperties": false,
      "required": ["overall_confidence"],
      "properties": {
        "overall_confidence": { "enum": ["very_low","low","medium","high","very_high"] },
        "notes": { "type": "string" }
      }
    }
  },

  "$defs": {
    "provenance": { "enum": ["user_stated","user_confirmed","literature","llm_domain","llm_default"] },
    "weight":     { "type": "number", "minimum": 0, "maximum": 1 },
    "ci":         { "type": "array", "items": { "type": "number" }, "minItems": 2, "maxItems": 2 },

    "belief": {
      "type": "object",
      "required": ["token"],
      "properties": { "token": { "type": "string" } },
      "oneOf": [
        {
          "title": "baseline",
          "additionalProperties": false,
          "properties": {
            "token": { "const": "baseline" },
            "level": { "type": "number", "minimum": 0, "maximum": 1 },
            "tilt":  { "type": "object", "additionalProperties": { "type": "number", "minimum": -1, "maximum": 1 } }
          }
        },
        {
          "title": "bump",
          "additionalProperties": false,
          "required": ["token", "dim", "mode", "weight", "provenance"],
          "properties": {
            "token": { "const": "bump" },
            "dim":   { "type": "string" },
            "mode":  { "type": "number" },
            "ci":    { "$ref": "#/$defs/ci" },
            "skew":  { "type": "number", "minimum": -1, "maximum": 1 },
            "sign":  { "enum": ["peak", "trough"] },
            "weight":     { "$ref": "#/$defs/weight" },
            "provenance": { "$ref": "#/$defs/provenance" },
            "note":  { "type": "string" }
          }
        },
        {
          "title": "trend",
          "additionalProperties": false,
          "required": ["token", "dim", "direction", "weight", "provenance"],
          "properties": {
            "token": { "const": "trend" },
            "dim":   { "type": "string" },
            "direction": { "enum": ["increase", "decrease"] },
            "knee":  { "type": "number" },
            "weight":     { "$ref": "#/$defs/weight" },
            "provenance": { "$ref": "#/$defs/provenance" },
            "note":  { "type": "string" }
          }
        },
        {
          "title": "threshold",
          "additionalProperties": false,
          "required": ["token", "dim", "at", "active", "weight", "provenance"],
          "properties": {
            "token": { "const": "threshold" },
            "dim":   { "type": "string" },
            "at":    { "type": "number" },
            "active": { "enum": ["above", "below"] },
            "sharpness": { "enum": ["soft", "sharp"] },
            "weight":     { "$ref": "#/$defs/weight" },
            "provenance": { "$ref": "#/$defs/provenance" },
            "note":  { "type": "string" }
          }
        },
        {
          "title": "periodic",
          "additionalProperties": false,
          "required": ["token", "dim", "period", "weight", "provenance"],
          "properties": {
            "token": { "const": "periodic" },
            "dim":   { "type": "string" },
            "period": { "type": "number", "exclusiveMinimum": 0 },
            "phase":  { "type": "number" },
            "weight":     { "$ref": "#/$defs/weight" },
            "provenance": { "$ref": "#/$defs/provenance" },
            "note":  { "type": "string" }
          }
        },
        {
          "title": "categorical",
          "additionalProperties": false,
          "required": ["token", "dim", "preferences", "weight", "provenance"],
          "properties": {
            "token": { "const": "categorical" },
            "dim":   { "type": "string" },
            "preferences": { "type": "object",
                             "additionalProperties": { "type": "number", "minimum": 0, "maximum": 1 } },
            "weight":     { "$ref": "#/$defs/weight" },
            "provenance": { "$ref": "#/$defs/provenance" },
            "note":  { "type": "string" }
          }
        },
        {
          "title": "joint_region",
          "additionalProperties": false,
          "required": ["token", "dims", "mode", "weight", "provenance"],
          "properties": {
            "token": { "const": "joint_region" },
            "dims":  { "type": "array", "items": { "type": "string" }, "minItems": 2 },
            "mode":  { "type": "object", "additionalProperties": { "type": "number" } },
            "ci":    { "type": "object", "additionalProperties": { "$ref": "#/$defs/ci" } },
            "weight":     { "$ref": "#/$defs/weight" },
            "provenance": { "$ref": "#/$defs/provenance" },
            "note":  { "type": "string" }
          }
        },
        {
          "title": "ridge",
          "additionalProperties": false,
          "required": ["token", "dims", "relation", "weight", "provenance"],
          "properties": {
            "token": { "const": "ridge" },
            "dims":  { "type": "array", "items": { "type": "string" }, "minItems": 2, "maxItems": 2 },
            "relation": { "enum": ["negative", "positive"] },
            "width": { "enum": ["narrow", "medium", "wide"] },
            "through": { "type": "object", "additionalProperties": { "type": "number" } },
            "weight":     { "$ref": "#/$defs/weight" },
            "provenance": { "$ref": "#/$defs/provenance" },
            "note":  { "type": "string" }
          }
        },
        {
          "title": "conditional",
          "additionalProperties": false,
          "required": ["token", "when", "then", "weight", "provenance"],
          "properties": {
            "token": { "const": "conditional" },
            "when":  { "type": "object", "additionalProperties": false,
                       "required": ["dim", "region"],
                       "properties": { "dim": { "type": "string" },
                                       "region": { "$ref": "#/$defs/ci" } } },
            "then":  { "type": "object" },
            "mode":  { "enum": ["gate", "synergy"] },
            "weight":     { "$ref": "#/$defs/weight" },
            "provenance": { "$ref": "#/$defs/provenance" },
            "note":  { "type": "string" }
          }
        },
        {
          "title": "plateau",
          "additionalProperties": false,
          "required": ["token", "region", "weight", "provenance"],
          "properties": {
            "token": { "const": "plateau" },
            "region": { "type": "object", "additionalProperties": { "$ref": "#/$defs/ci" } },
            "weight":     { "$ref": "#/$defs/weight" },
            "provenance": { "$ref": "#/$defs/provenance" },
            "note":  { "type": "string" }
          }
        },
        {
          "title": "avoid",
          "additionalProperties": false,
          "required": ["token", "weight", "provenance"],
          "properties": {
            "token": { "const": "avoid" },
            "region": { "type": "object", "additionalProperties": { "$ref": "#/$defs/ci" } },
            "dim":  { "type": "string" },
            "mode": { "type": "number" },
            "ci":   { "$ref": "#/$defs/ci" },
            "weight":     { "$ref": "#/$defs/weight" },
            "provenance": { "$ref": "#/$defs/provenance" },
            "note":  { "type": "string" }
          }
        },
        {
          "title": "anchors",
          "additionalProperties": false,
          "required": ["token", "points"],
          "properties": {
            "token": { "const": "anchors" },
            "bandwidth": { "type": "number", "exclusiveMinimum": 0 },
            "provenance": { "$ref": "#/$defs/provenance" },
            "note":  { "type": "string" },
            "points": {
              "type": "array", "minItems": 1,
              "items": {
                "type": "object",
                "additionalProperties": false,
                "required": ["coords", "value"],
                "properties": {
                  "coords": { "type": "object" },
                  "value":  { "type": "number", "minimum": 0, "maximum": 1 }
                }
              }
            }
          }
        },
        {
          "title": "smoothness",
          "additionalProperties": false,
          "required": ["token"],
          "properties": {
            "token": { "const": "smoothness" },
            "level": { "enum": ["very_smooth","smooth","medium","rough","very_rough"] },
            "per_dim": { "type": "object",
                         "additionalProperties": { "enum": ["very_smooth","smooth","medium","rough","very_rough"] } }
          }
        },
        {
          "title": "noise",
          "additionalProperties": false,
          "required": ["token", "level"],
          "properties": {
            "token": { "const": "noise" },
            "level": { "enum": ["low","medium","high","very_high"] },
            "repeats_recommended": { "type": "boolean" }
          }
        },
        {
          "title": "modality",
          "additionalProperties": false,
          "required": ["token"],
          "properties": {
            "token": { "const": "modality" },
            "expectation": { "enum": ["unimodal","few","many"] },
            "n_optima": { "type": "integer", "minimum": 1 }
          }
        }
      ]
    }
  }
}
```

The schema pins *shape*. It cannot check three things that are still your job, and that
§7's checklist re-lists: (a) coordinates lie inside their dimension's `range`/`values`;
(b) exactly one shaping belief has `weight: 1.0`; (c) the document is *faithful* to what
the expert actually said.

---

## 2. The schema (typed search space)

`schema` has two fields: `objective` (`"maximize"` | `"minimize"`),
and `dimensions` (a list of dimension specs). Each dimension:

```json
{
  "name":    "x1",                          // unique, snake_case; beliefs refer to this
  "type":    "continuous" | "ordinal" | "categorical", // REQUIRED; determines which fields below are required
  "range":   [0, 100],                      // REQUIRED; for continuous: [low, high]
  "values":  [1, 2, 4, 8],                  // REQUIRED; for ordinal (ordered) & categorical (unordered)
  "scale":   "linear" | "log",             // continuous only; default "linear"
  "meaning": "free-text description of the knob"  // optional, aids elicitation
}
```

Rules:

- **continuous**: give `range: [low, high]`. If the quantity naturally lives on a
  multiplicative axis (a knob spanning several orders of magnitude, where a *ratio* is
  more meaningful than a *difference*), set `scale: "log"` — then all coordinates for that
  dimension (modes, intervals, thresholds) are read on the log axis, so a `mode` of `1e-3`
  sits exactly between `1e-4` and `1e-2`. Use `linear` otherwise. `log` requires
  `range[0] > 0`.
- **ordinal**: a finite, *ordered* set (`values` in increasing order). Shape tokens
  (`trend`, `threshold`, `bump`, `periodic`) may be used; coordinates are one of the
  listed values.
- **categorical**: a finite, *unordered* set. Only the `categorical` token (§4.6) applies;
  shapes do not, because unordered levels have no geometry.

**Hard limits are part of the schema, not a belief.** A knob's hard operating bounds go
directly in its `range`/`values` here — the optimizer never searches outside the schema,
so safety and feasibility limits (a knob that must never exceed a value) belong in the
dimension itself, not in the belief list. Feasibility that can only be discovered by
*measuring* is handled separately by the optimizer's constrained-search machinery, also
outside this document. The belief list is only ever about *where the optimum is likely to
be* — and by design nothing in it can remove a point from the search, so a wrong belief
can only slow the search, never make the true optimum unreachable.

---

## 3. Metadata every belief carries

Every entry in `beliefs` **except the meta tokens (§4.13–4.15) and `anchors`** carries
exactly two metadata fields plus an optional note:

```json
"weight":     0.0-1.0,          // REQUIRED; relative strength; the strongest region = 1.0
"provenance": "user_stated" | "user_confirmed" | "literature" | "llm_domain" | "llm_default", // REQUIRED
"note":       "..." // OPTIONAL: a citation, a quote, or a one-line justification
```

The three quantities are kept orthogonal and each has exactly one home:

| quantity | question it answers | where it lives |
|---|---|---|
| **strength** | how much does this region matter *relative to the others*? | `weight` |
| **location uncertainty** | how *precisely* is this region located? | the interval `ci` (wider = less sure where) |
| **overall trust** | how much should the optimizer believe the document at all? | `trust.overall_confidence` × per-belief `provenance` |

**Provenance** (who is the source — this scales how much the compiler trusts the belief):

| value | meaning |
|---|---|
| `user_stated` | the human expert asserted it directly |
| `user_confirmed` | you proposed it and the human accepted it |
| `literature` | drawn from a cited source (put the citation in `note`) |
| `llm_domain` | your own parametric domain knowledge, unverified |
| `llm_default` | a structural guess you made to fill a gap |

Be honest with provenance: it is the audit trail, and the compiler down-weights guesses
relative to human statements when it computes `α`.

---

## 4. THE VOCABULARY

Tokens are organized by the layer of structure they express (baseline → per-dimension
shapes → interactions → region operators → the anchor escape hatch → meta). Tokens marked
**‡** form a mathematically complete basis on their own: `bump`, `joint_region`, and
`anchors` can, in principle, express *any* landscape. The other tokens are **macros** —
they say a common expert intuition in *one* token that would otherwise cost many. Prefer a
macro when it fits the expert's own words ("more is better", "avoid this band"); fall back
to `anchors` only for things the expert can *point to* but not *name*.

Coordinates are always in the dimension's **native units**. `weight` and `provenance` are
as in §3 and omitted from the per-token snippets below for brevity.

**Credible-interval convention (applies to every `ci` in this section):** a `ci` is a
**90% credible interval** — the range you would give 90% odds contains the best value.
The compiler reads its half-width as 1.645 σ, so a *tight* `ci` ⇒ a *sharp* peak, a *wide*
`ci` ⇒ a *broad* hill. Always inside the dimension's `range`.

### Layer 0 — Baseline

#### 4.1 `baseline` — the default level and any global tilt (the μ term)

```json
{ "token": "baseline", "level": 0.1, "tilt": { "x1": 0.3 } }
```

- `level` (0..1, default `0.1`): the floor value everywhere before local structure.
- `tilt` (optional): a gentle global slope, `{ dimension_name: slope }` with each slope in
  `[-1,1]`, e.g. `"tilt": { "x1": 0.3 }` means "all else equal, higher `x1` is slightly
  better across the whole space." Omit if the surface has no overall drift.
- Use **at most one** `baseline`. Carries no `weight`/`provenance`. If omitted, the
  compiler assumes a small flat baseline.

### Layer 1 — Per-dimension marginal shapes (the `g_i`)

For **continuous / ordinal** dimensions only.

#### 4.2 `bump` **‡** — a single peak (or trough) on one dimension

```json
{ "token": "bump", "dim": "x1", "mode": 80,
  "ci": [60, 110], "skew": 0.0, "sign": "peak" }
```

- `mode`: the most-likely-best value (native units).
- `ci`: the 90% credible interval `[low, high]` (native units).
- `skew` (optional, `-1..1`, default `0`): asymmetric falloff; negative = falls off faster
  above the mode.
- `sign` (optional, default `"peak"`): `"trough"` makes it a *dip* (subtractive) instead.
- Captures: *"the best value of this knob is around 80, give or take"*.

#### 4.3 `trend` — monotone increase/decrease, with an optional saturation knee

```json
{ "token": "trend", "dim": "x1", "direction": "increase", "knee": 80 }
```

- `direction`: `"increase"` (bigger is better) or `"decrease"` (smaller is better).
- `knee` (optional, native units): the value past which it stops mattering (saturates
  flat). Omit for a pure ramp across the whole range.
- Captures: *"bigger is better, then it plateaus"*, *"smaller is better"*.
- **Use `trend` (not `bump`) for a *vague directional lean*.** When the expert says "I
  think something small/large but I'm not sure" — a preference for one end of the range
  *without a definite peak* — use a low-`weight` `trend` (e.g. `direction: "decrease"` for
  "lean small"). A `bump` would wrongly commit to a peak at a specific value.

#### 4.4 `threshold` — a regime change / near-step at a boundary value

```json
{ "token": "threshold", "dim": "x1", "at": 50, "active": "above", "sharpness": "sharp" }
```

- `at`: the boundary value (native units).
- `active`: which side is *good* — `"above"` or `"below"` the boundary.
- `sharpness` (optional, default `"soft"`): `"sharp"` = near-discontinuous switch,
  `"soft"` = gradual transition.
- Captures: *"nothing happens below a cutoff value, then the response switches on"*.

#### 4.5 `periodic` — oscillatory structure on one dimension

```json
{ "token": "periodic", "dim": "x1", "period": 200, "phase": 0 }
```

- `period` (native units, > 0): distance between successive peaks.
- `phase` (optional, native units, default `0`): location of the first peak.
- Captures: *"the response peaks at regular intervals along this knob"*. **Use rarely** —
  only for genuine oscillation. (It is a full member of the vocabulary because describing
  real oscillation with bumps costs one bump *per period*; but genuine periodic beliefs
  are uncommon, so reach for it only when the expert truly means "peaks every N units.")

#### 4.6 `categorical` — preference weights over the levels of an unordered dimension

```json
{ "token": "categorical", "dim": "c1",
  "preferences": { "level_a": 1.0, "level_b": 0.6, "level_c": 0.4 } }
```

- `preferences`: a value in `[0,1]` for the levels you have an opinion on. This is a
  *multiplicative gate*: it scales the plausibility of all points sharing that level.
- Captures: *"one level usually beats the others"*. Omit a level ⇒ treated as neutral
  (`1.0`), so list every level you have an opinion on.

### Layer 2 — Interactions (the `h_ij`, pairwise)

#### 4.7 `joint_region` **‡** — a localized optimum living in a subspace

```json
{ "token": "joint_region", "dims": ["x1", "x2"],
  "mode": { "x1": 150, "x2": 40 },
  "ci":   { "x1": [120, 180], "x2": [30, 55] } }
```

- Like `bump`, but the peak lives jointly in **two** dimensions at once (a 2-D hill).
- `mode` and `ci` are given **per named dimension** (each `ci` a 90% interval).
- Captures: *"the sweet spot is high `x1` **and** low `x2` together"*.
- **Never span more than 2 dims with one `joint_region`.** The compiled joint shape is a
  *product* of per-dimension Gaussians, so its mass shrinks exponentially with the number
  of dimensions: over 5+ dims it becomes a needle of vanishing volume that expresses no
  usable preference anywhere else in the box. To say *"the optimum is near point p"* in a
  d-dimensional space, write **one `bump` per dimension at `p_i`** (per-dimension shapes
  are ADDED, so each coordinate keeps global influence along its own axis — see §8.1's
  amplitude bump). Reserve `joint_region` for a genuine pairwise coupling.

#### 4.8 `ridge` — a trade-off line through two dimensions (correlation)

```json
{ "token": "ridge", "dims": ["x1", "x2"], "relation": "negative",
  "width": "medium", "through": { "x1": 2.0, "x2": 100 } }
```

- `relation`: `"negative"` = as one goes up the other should go down; `"positive"` = they
  rise together.
- `width` (optional, default `"medium"`): `"narrow"|"medium"|"wide"` — how tightly the good
  region hugs the line.
- `through` (optional): a point the ridge passes through; if omitted it runs through the
  center.
- Captures: *"more `x1` needs proportionally less `x2`"*.
- **`ridge` vs `conditional`:** use `ridge` when the two knobs *co-vary smoothly* along a
  trade-off (a diagonal good region). Use `conditional` when one knob *gates* the other in
  a regime sense. Smooth trade-off → `ridge`; on/off or regime-switch → `conditional`.

#### 4.9 `conditional` — context-dependence: dim B's shape activates within a region of dim A

```json
{ "token": "conditional",
  "when": { "dim": "x1", "region": [50, 200] },
  "then": { "token": "bump", "dim": "x2", "mode": 100, "ci": [60, 140] },
  "mode": "gate" }
```

- `when`: the region of the gating dimension that turns the effect on.
- `then`: a **nested** single belief token (a `bump`, `trend`, `threshold`, or `plateau`
  on another dimension) that applies only where `when` holds.
- `mode` (optional, default `"gate"`): `"gate"` = the `then` shape is added only inside the
  `when` region; `"synergy"` = the `then` shape multiplies the whole score (use for strong
  "only matters when…" couplings).
- Captures: *"`x2` only matters when `x1` is high"*.

### Region operators (sign/mask over any subspace)

#### 4.10 `plateau` — an indifference region: everything in this band is equally fine

```json
{ "token": "plateau", "region": { "x1": [380, 420] } }
```

- `region`: `[low, high]` per named dimension (one or more dims). Raises `π` *flatly* over
  the box.
- Captures: *"anywhere in this band of `x1` is equally good"*.

#### 4.11 `avoid` — a penalized (soft-bad) region

```json
{ "token": "avoid", "region": { "x1": [480, 500] } }
```

- `region`: box per named dimension, **or** use `dim`/`mode`/`ci` for a bump-shaped
  penalty.
- Captures: *"this region tends to give poor results — steer away"*. Subtractive; the
  point stays searchable, so a wrong `avoid` can only slow the search, never remove the
  optimum.
- **`avoid` is for bad-but-measurable regions.** Use it whenever a region is wasteful or
  poor — even "don't bother," "it diverges," "never works" — because the point is still
  worth being able to reach. A *hard* limit that must never be evaluated is **not** a
  belief: put it in the dimension's `range`/`values` in the schema (§2).

### The escape hatch (completeness)

#### 4.12 `anchors` **‡** — specific points the expert can point to

```json
{ "token": "anchors", "points": [
    { "coords": { "x1": 42, "x2": 1.0,  "c1": "level_a" }, "value": 0.95 },
    { "coords": { "x1": 88, "x2": 0.01, "c1": "level_c" }, "value": 0.1  }
] }
```

- Each point: `coords` (a value for **every** dimension, categorical as the level label)
  and a `value` in `[0,1]` (`1.0` = known-great, `0.0` = known-bad).
- **Anchors are exact.** The compiler applies anchors *last, after normalizing the rest of
  the map*: at a candidate that coincides with an anchor, `π` is set to *exactly* that
  anchor's `value`; between anchors it blends smoothly (a compact-support partition of
  unity); away from all anchors the rest of the belief document is left untouched. This is
  what lets a pure-anchor document reproduce any table of values exactly.
- `bandwidth` (optional, default `0.15` in normalized units): how far each anchor's
  influence reaches. Carries **no `weight`** (v2); provenance is optional on the token.
- This token can express **anything** describable — use it when no shape/interaction macro
  fits ("I just know these three configs and how they did").

### Layer 3 — Regularity / meta (these set the OPTIMIZER, not the map)

These tokens do **not** shape `π`; they tune the GP surrogate and search schedule. They
carry no `weight`/`provenance`. Include them when the expert has said something about *how
the landscape behaves*, not *where the optimum is*.

#### 4.13 `smoothness` — how gradually the surface changes → GP lengthscale ρ

```json
{ "token": "smoothness", "level": "smooth" }
```

- `level`: `"very_smooth" | "smooth" | "medium" | "rough" | "very_rough"`.
- Or per dimension: `{ "token": "smoothness", "per_dim": { "x1": "smooth", "x2": "rough" } }`.
- Captures: *"performance changes gradually along `x1`"* vs *"tiny changes flip the
  result"*.

#### 4.14 `noise` — how repeatable a single measurement is → GP noise floor

```json
{ "token": "noise", "level": "high", "repeats_recommended": true }
```

- `level`: `"low" | "medium" | "high" | "very_high"`.
- `repeats_recommended` (optional): `true` if each measurement is so noisy the optimizer
  should consider repeat queries.
- Captures: *"the same config gives noticeably different scores run to run"*.

#### 4.15 `modality` — how many competing optima to expect → exploration κ and initial budget

```json
{ "token": "modality", "expectation": "few" }
```

- `expectation`: `"unimodal"` (one optimum), `"few"` (a handful), or `"many"` (rugged).
- Or `{ "token": "modality", "n_optima": 3 }`.
- Captures: *"there are probably several separate good regions"* → more exploration.

---

## 5. Trust block

```json
"trust": {
  "overall_confidence": "medium",
  "notes": "expert knows the process well, but this particular setup is new to them"
}
```

- `overall_confidence`: one of the five confidence words — the expert's global self-rating.
  The compiler combines this with each belief's `provenance` (and its `weight`) to produce
  the single trust scalar `α` (clamped to a safe `[0.1, 0.9]` band, so the shaped
  multiplier's worst-case constant is at most 10).
- `notes` (optional): free text for the audit trail.

---

## 6. How the compiler will read your document (mental model, not your job)

You do not compute anything, but knowing the compile rule helps you write faithfully:

```
score(x) = baseline(x)
         + Σ (additive shapes: bump-peak, trend, threshold, periodic,
              joint_region, ridge, plateau, conditional-gate)   × their weights
         − Σ (avoid, bump-trough)                               × their weights
score(x) = score(x) × Π (gates: categorical, conditional-synergy)
π(x)     = normalize( max(score(x), 0) )  →  [0,1]
π(x)     = blend π(x) toward anchor values   # anchors applied LAST, after normalization;
                                             #   exact at anchored candidates (§4.12)
```

- Because the shape/interaction layer is **normalized before anchors are applied**, only
  *relative* weights and *shapes* matter there — never absolute magnitudes. Anchors are
  the one exception: their `value` is an absolute target in `[0,1]`, honored exactly.
- No belief token can zero out or remove a candidate — the closest, `avoid`, only subtracts
  and is re-normalized, so `π` stays ≥ 0 and every point remains reachable. Hard
  feasibility lives in the schema/optimizer, not here.
- The three **‡** tokens plus interpolation are provably enough to express any landscape;
  the macros exist so you can say common patterns compactly. Prefer the macro that matches
  the expert's phrasing.
- Keep interactions to **pairs** (`joint_region`, `ridge`, `conditional`). If a genuine
  three-way effect exists, express it with `anchors`. A `joint_region` spanning many
  dimensions compiles to a near-zero-volume product needle — a believed optimum
  *location* is a set of per-dimension `bump`s, one per coordinate, never a wide
  `joint_region`.
- The meta tokens (`smoothness`/`noise`/`modality`) bypass this rule entirely — they set GP
  hyperparameters, not `π`.

---

## 7. Validity checklist (self-check before you output)

- [ ] Exactly the three top-level keys: `schema`, `beliefs`, `trust`.
- [ ] Every `dim`/`dims` in a belief names a dimension that exists in `schema.dimensions`.
- [ ] Every coordinate (`mode`, `ci`, `at`, `region`, `through`, anchor `coords`) is in
      native units and inside that dimension's `range`/`values`.
- [ ] Every `ci` is a **90% credible interval** (not a ±1σ or a hard bound).
- [ ] Continuous dims that are multiplicative use `scale: "log"`.
- [ ] Exactly one shaping belief has `weight: 1.0` (the strongest region); others smaller.
- [ ] `categorical` only on categorical dims; shape tokens only on continuous/ordinal dims.
- [ ] `preferences` lists a value for the levels you have an opinion on.
- [ ] Hard/unsearchable limits are in the schema's `range`/`values`, not in a belief.
- [ ] Every non-meta, non-`anchors` belief has `weight` and `provenance` — and **no**
      per-belief `confidence` field.
- [ ] No `joint_region` (or `ridge`/`conditional`) spanning more than 2 dimensions — a
      believed optimum location in d dims is d per-dimension `bump`s at its coordinates.
- [ ] No invented tokens — anything unnameable is expressed with `anchors`.
- [ ] Output is a single valid JSON object and nothing else.

---

## 8. Worked examples (study these — they are worth more than the checklist)

Three faithful documents, each showing a different situation. The middle one is the case
where beginners most often go wrong: a **log-scale** dimension.

### 8.1 Neurostimulation grid (2D spatial + amplitude)

*Expert said:* "With this array the strong responses are usually rostral and lateral —
around column 6, row 3. Going too deep in amplitude past about 250 µA just recruits the
wrong muscle. Below rheobase, roughly 40 µA, nothing happens at all. The map is pretty
smooth but every reading is noisy, so repeat if you can."

```json
{
  "schema": {
    "objective": "maximize",
    "dimensions": [
      { "name": "rostrocaudal", "type": "ordinal", "values": [0,1,2,3,4,5,6,7],
        "meaning": "electrode column, rostral->caudal" },
      { "name": "mediolateral", "type": "ordinal", "values": [0,1,2,3],
        "meaning": "electrode row, medial->lateral" },
      { "name": "amplitude", "type": "continuous", "range": [10, 300], "scale": "linear",
        "meaning": "stimulation current in microamps" }
    ]
  },
  "beliefs": [
    { "token": "baseline", "level": 0.1 },
    { "token": "joint_region", "dims": ["rostrocaudal", "mediolateral"],
      "mode": { "rostrocaudal": 6, "mediolateral": 3 },
      "ci":   { "rostrocaudal": [5, 7], "mediolateral": [2, 3] },
      "weight": 1.0, "provenance": "user_stated",
      "note": "strong responses are usually rostral and lateral" },
    { "token": "threshold", "dim": "amplitude", "at": 40, "active": "above",
      "sharpness": "sharp", "weight": 0.7, "provenance": "user_stated",
      "note": "rheobase ~40 uA; nothing below it" },
    { "token": "avoid", "region": { "amplitude": [250, 300] },
      "weight": 0.6, "provenance": "user_stated",
      "note": "too deep recruits the wrong muscle" },
    { "token": "smoothness", "level": "smooth" },
    { "token": "noise", "level": "high", "repeats_recommended": true }
  ],
  "trust": {
    "overall_confidence": "high",
    "notes": "experienced with this array and prep"
  }
}
```

Note: the spatial belief is a `joint_region` (rostral **and** lateral *together*), not two
separate `bump`s. The 250 µA limit is a soft `avoid` (the point is bad but still worth
being able to reach), **not** a schema bound — the operator can physically go there.

### 8.2 ML hyperparameters — the log-scale case (read carefully)

*Expert said:* "Learning rate around 1e-3 usually works, maybe anywhere from 3e-4 to 3e-2;
above ~5e-2 it diverges. Bigger batches train faster and are usually at least as good.
Adam is my default; SGD rarely wins here."

```json
{
  "schema": {
    "objective": "maximize",
    "dimensions": [
      { "name": "learning_rate", "type": "continuous", "range": [1e-5, 1e-1],
        "scale": "log", "meaning": "optimizer step size" },
      { "name": "batch_size", "type": "ordinal", "values": [16, 32, 64, 128, 256] },
      { "name": "optimizer", "type": "categorical", "values": ["adam", "sgd", "rmsprop"] }
    ]
  },
  "beliefs": [
    { "token": "baseline", "level": 0.1 },
    { "token": "bump", "dim": "learning_rate", "mode": 1e-3, "ci": [3e-4, 3e-2],
      "weight": 1.0, "provenance": "user_stated",
      "note": "'around 1e-3, maybe 3e-4 to 3e-2' — 90% interval on the LOG axis" },
    { "token": "avoid", "dim": "learning_rate", "mode": 1e-1, "ci": [5e-2, 1e-1],
      "weight": 0.5, "provenance": "user_stated", "note": "diverges above ~5e-2" },
    { "token": "trend", "dim": "batch_size", "direction": "increase",
      "weight": 0.5, "provenance": "user_stated", "note": "'bigger trains faster, >= as good'" },
    { "token": "categorical", "dim": "optimizer",
      "preferences": { "adam": 1.0, "rmsprop": 0.6, "sgd": 0.4 },
      "weight": 0.7, "provenance": "user_stated", "note": "Adam default, SGD rarely wins" }
  ],
  "trust": {
    "overall_confidence": "medium",
    "notes": "solid intuition, but this dataset is new to the expert"
  }
}
```

The trap: because `learning_rate` is `scale: "log"`, the interval `[3e-4, 3e-2]` is read on
the **log axis** — its half-width in log-space is what becomes σ (÷1.645), so `1e-3` sits
naturally near the center. Give the mode and interval in **native µ-units** (`1e-3`,
`3e-4`), *never* pre-log them; the compiler does the transform. A common mistake is
writing a symmetric-in-linear-space interval like `[0, 2e-3]` around `1e-3` — on a log axis
`0` is not even representable. Give the range the way the expert says it.

### 8.3 Sparse knowledge — anchors only

*Expert said:* "I don't have a theory, but I've run three configs. The first was excellent,
the second was useless, the third was so-so."

```json
{
  "schema": {
    "objective": "maximize",
    "dimensions": [
      { "name": "temperature", "type": "continuous", "range": [20, 200], "scale": "linear" },
      { "name": "catalyst",    "type": "categorical", "values": ["pd", "ni", "cu"] }
    ]
  },
  "beliefs": [
    { "token": "anchors", "provenance": "user_stated", "points": [
        { "coords": { "temperature": 150, "catalyst": "pd" }, "value": 0.95 },
        { "coords": { "temperature": 40,  "catalyst": "cu" }, "value": 0.05 },
        { "coords": { "temperature": 110, "catalyst": "ni" }, "value": 0.5  }
    ] }
  ],
  "trust": {
    "overall_confidence": "medium",
    "notes": "three observed configs, no mechanistic prior"
  }
}
```

Each of those three grid points will compile to *exactly* `0.95`, `0.05`, `0.5`; the space
between them is smoothly interpolated. This is the "point-to-it, can't-name-it" escape
hatch — reach for it only when no shape/interaction macro fits.