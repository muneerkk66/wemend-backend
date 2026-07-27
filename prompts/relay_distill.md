# Relay distill prompt — Gemma 4 E4B

The shuttle step: takes what one partner said in private, produces what the app
says aloud to the other. **This is the core of the product.** Everything else is
plumbing.

Validated against `gemma4:e4b` (Q4_K_M) on an RTX 4090 — 114 tok/s, valid JSON.

## Call it with `format: "json"`

Ollama's `format: "json"` is required. Without it Gemma wraps output in
markdown fences despite being told "ONLY JSON", and the parse fails.

```json
{
  "model": "gemma4:e4b",
  "format": "json",
  "options": { "temperature": 0.2, "num_ctx": 8192 },
  "messages": [ { "role": "system", "content": "<below>" }, ... ]
}
```

## System prompt

Substitute real names for `{SPEAKER}` / `{LISTENER}` — see why below.

```
You relay messages between two partners in conflict.

SPEAKER = {SPEAKER}. LISTENER = {LISTENER}.
You will hear what {SPEAKER} said privately. You will then SPEAK ALOUD TO {LISTENER}.

Rules for the relay text:
1. Write it as YOU (the mediator) speaking TO {LISTENER} ABOUT {SPEAKER}. Use
   "{SPEAKER}" and "he"/"she", never "I" or "you" for {SPEAKER}. Never swap who
   did what.
2. KEEP the specific concrete behaviour {SPEAKER} named. Do not generalise it away.
3. KEEP any statement about withdrawing, giving up, or leaving. Never drop it.
4. Remove insults, contempt, and absolutes (never/always).

Return JSON keys: relay, underlying_need, concrete_behaviour,
withdrawal_signal (boolean), abuse_flag (boolean), reason.
```

## Why each rule exists — three real failure modes

Measured, not hypothetical. A first draft that only said *"warm, neutral, strips
contempt but preserves the grievance"* failed all three ways on the first test input
(*"She never bloody listens… glued to that phone… I work twelve hours and come home
to a zombie… wondering why I even bother coming home."*).

**1. Perspective flip — the dangerous one.** Output was
*"When **you** get home after a long day, **I** feel really disconnected from you."*
Spoken to the wife, that asserts the husband said the opposite of what he said. An
app that misattributes statements between partners in conflict doesn't just fail —
it manufactures new conflict. Rule 1 fixes it; **name the speakers explicitly**,
because "husband"/"wife" alone still flipped.

**2. Concrete behaviour dissolved into therapy-speak.** "Glued to that phone"
became "disconnected". Vague feelings give the listener nothing to act on — the
specific behaviour is what makes a grievance addressable. Rule 2.

**3. The most serious signal was silently dropped.** *"Wondering why I even bother
coming home"* vanished entirely. That is the highest-stakes thing in the input — a
withdrawal marker — and softening removed it. Rule 3 plus the explicit
`withdrawal_signal` field forces it to survive.

Failure mode 3 generalises: **any purely subtractive instruction ("remove contempt")
will eventually remove the signal too.** Every must-preserve needs its own named
output field, so dropping it is a visibly missing value rather than a silent edit.

## Known remaining gap

Rule 4 is the least reliably followed — "every night" survived into the relay as an
absolute. Low severity, but worth an eval case.

## Do not ship without

- **An eval set.** These three failures were caught by *one* hand-written input.
  Build a fixture set of conflict utterances with expected `abuse_flag` /
  `withdrawal_signal` values and assert on every prompt change.
- **A perspective-flip assertion.** Reject any `relay` containing first-person
  ("I feel", "I am") for the speaker — cheap regex, catches failure mode 1.
- **Consent gate.** The speaker approves `relay` before it is voiced. See
  `ARCHITECTURE.md`.
- **`abuse_flag` must not be trusted alone.** A 4B model's boolean is a triage
  hint, not a safety control. Route any true to a human escalation path.
