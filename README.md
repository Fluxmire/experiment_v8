# Three.js reconstruction miner: merged v8

Image-to-Three.js pipeline for the 404-GEN GPT competition. Root `main.py` exposes
`async def generate(image: bytes, seed: int, witness: AsyncWitnessClient) -> str | None`.
- **Our code:** the pipeline is `miner/merge.py`; the system prompt is `miner/rewrite_prompts.py` (with our LIGHT block).
- **Base:** everything else is `Olauge-Secret/sedge-wren@fc79ce3`, unchanged (`provenance.json`).

**Status: offline-tested only** (control flow, request ids, budget, gates). Not yet run live or judged.

## What changed vs v7, and why
These come from round-3 source reviews: https://github.com/Fluxmire/round3-analysis (private). Ideas only; no
competitor code is copied.

1. **Gated refines.** Each lineage gets **3 Sol-high post refines**, and each refine must pass a gate against its
   parent.
   - **The gate:** 2 Sol-low calls (both presentation orders) in the judge's rubric, on the 4 front views both models
     already rendered for the check. No extra renders.
   - **Pro-child:** the child stays unless the parent scores better by more than 0.5, so only clear regressions are
     reverted, and v5's depth gains survive judge noise.
   - **Why:** every top round-3 cluster gates each step against the parent (Olauge `each_lineage`, phuy
     `per_lineage`); v7 accepted any refine that rendered.
2. **Neutral-light rules** in the system prompt, our own wording. The judge has rendered in bright neutral light since
   2026-11-05, and the rules follow from that: keep darks dark, carry a metal's brightness in its colour, build the
   underside/top/back, real glass, nothing the photo does not show (no floating parts or halos).
3. **Output caps = reservations from measured maxima:** Astra 22k, Sol 26k (loops 1–9 maxima: 16,529 / 19,716 over
   657 / 1,645 calls). Smaller worst-case reservations let more refines fit the budget (lark-ridge's lesson).
4. **Duel:** a view counts only when both orders replied. No usable view keeps lineage 1.

## Pipeline (about 32 model calls per image)
1. **Two lineages, run concurrently** (sedge's chains):
   - Astra-medium / Sol-medium draft, ≤2 repairs.
   - Sol-high → Astra-high rewrites.
   - **3 gated Sol-high post refines.**
   - sedge's checked finish.
   - Every rewrite shows the SIDES sheet (right / back / left / top).
2. **Final pick:** an 8-call copy of the judge's stage 1 (4 front views at 1024 px × both orders, Sol low) between the
   two finished lineages; a tie keeps lineage 1.

## Budget, determinism, 24 h
- **Fixed shares:** 2 × 2.25M credits / 24 calls per lineage, plus 8 × 62.5K / 1 per duel call = **5.0M / 56**.
- **Deterministic:** each part's guard sees only its own spend.
- **Per image:** about 3.4M credits expected; ≈ 2,960 model-s; ≈ 26 renders (≈ 3,400 per 128-prompt round, cap 10k).
- **128 images:** ≈ 17.5 h at 6 concurrent calls, ≈ 8.8 h at 12.

| | v5 (`experiment_v7`) | v7 (`experiment_v71`) | v8 |
|---|---|---|---|
| Lineages kept to the end | no (pick, then 3 refines) | yes, 2 refines each | yes, **3 gated refines each** |
| Final pick | Astra-low panel (1 call) | judge-copy duel (8) | judge-copy duel (8, both orders required) |
| Neutral-light rules | no | no | **yes** |
| Calls per image | ≈ 10 | ≈ 18 | ≈ 32 |
