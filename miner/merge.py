"""Merged v8 (Claude research): v7 + a parent gate on every lineage refine (3 Sol-high post refines per lineage),
neutral-light rules in the system prompt, output caps/reservations from measured maxima, and a duel that counts a view
only when both orders replied. v7 docstring follows.
Merged v7 (Claude research): v5 with two further ideas:
each of sedge's chains is a lineage kept to the end (own post refines + finish), and a judge-faithful 8-call stage-1 duel
picks between the two finished lineages (replacing sedge's single Astra-low pick). See README.md.
The v5 docstring follows. Merged v5 (Claude research, round 2 loops 7+8): loop-6 pipeline + a 3rd post refine + a per-image render cache.
Loop 6 docstring follows. Merged v4 + concurrent chains (Claude research, round 2 loop 6): sedge-wren backbone + newvictor's side-view rewrite feedback.
Evidence: round_2/loop_5/idea.md.

Backbone: Olauge-Secret/sedge-wren@fc79ce3 FLOW="lumen" (miner/lumen.py): Astra-medium and Sol-medium chains, each
draft repaired, then rewritten Sol-high -> Astra-high; Astra-low pick; Sol-high and Astra-high post refines; finish
(wrap + flat-object tilt). Same models, efforts, prompts, pick, request ids and budget guard as lumen.py.

Merged from SmoothTight/newvictor@0f4bc93 (miner404/pipeline.py SIDES and prompts.REWRITE item 5): every rewrite turn
(chain rewrites and post refines) additionally shows the current module's SIDES sheet: right, back and left at eye level
and the top (theta 90/180/270/0, phi 0/0/0/-90), 256 px each on white with neutral light, as one 2x2 image, with
newvictor's instruction to keep every part recognizable and complete any back or side that is missing, flat or hollow.
The front views, refine text and code block are sedge's, unchanged and in the same order.

Robustness (not in sedge): a dropped connection (APIConnectionError) is resent with the SAME request id up to
TRANSPORT_RETRIES times instead of failing the whole image.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re

import openai
from loguru import logger
from witness_client import AsyncWitnessClient, RenderOptions

from miner import images, lumen, pipeline, rewrite
from miner.rewrite import Calls, check, code_of, finish, first_module, image_part, refine_content, text_part

SIDES = RenderOptions(
    thetas=(90.0, 180.0, 270.0, 0.0),
    phis=(0.0, 0.0, 0.0, -90.0),
    img_size=256,
    cam_radius=2.0,
    cam_fov_deg=49.1,
    gap=images.GRID_GAP,
    bg_color="ffffff",
    lighting="neutral",
)
SIDES_LABEL = (
    "The same module seen from the other sides (one 2x2 image: top-left from the right, top-right from the back, "
    "bottom-left from the left, bottom-right from the top). Every part must stay recognizable from these sides too: "
    "complete any back or side that is missing, flat or hollow, without changing what already matches the photo in "
    "the front views:"
)
TRANSPORT_RETRIES, TRANSPORT_WAITS = 3, (5, 15, 45)
CHAIN_K0, POST_K0 = 40, 80
# Each chain's budget guard: own spend + this reserve (for the other chain and the post stage) + next worst case <= 5M.
# Loops 7/8: max chain spend 2.10M, chains combined 2.83M, so neither this nor the earlier shared guard ever bound.
CHAIN_RESERVE_CREDITS, CHAIN_RESERVE_CALLS = 1_000_000, 20
# sedge's post refine text already says "step k of 3" (lumen.POST_STEPS = 3) but runs 2 steps; v5 runs the 3rd:
POST_STAGES = (*lumen.POST_STAGES, (pipeline.MODEL, 3, ""))  # Sol high, "REFINEMENT STEP 3 of 3", with the sides sheet


class CachedWitness:
    """The witness with a per-image render cache: the renderer is deterministic, so an identical (source, options)
    pair is rendered once. Failures are not cached. Everything else passes through."""

    def __init__(self, witness: AsyncWitnessClient) -> None:
        self._w, self._cache, self.hits = witness, {}, 0

    async def render(self, source: str, options: RenderOptions = None):  # type: ignore[assignment]
        key = (hashlib.sha256(source.encode()).hexdigest(), repr(options))
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        out = await (self._w.render(source) if options is None else self._w.render(source, options))
        self._cache[key] = out
        return out

    def __getattr__(self, name):
        return getattr(self._w, name)  # Calls.k bases: chain c uses k from 40*c (ids 4001+ for chain 2), pick/post from 80


class Spend:
    """One image's spend, shared by the concurrent chains' Calls (sedge's guard reads it before every call)."""

    def __init__(self) -> None:
        self.credits, self.calls = 0.0, 0


class RCalls(Calls):
    """sedge's Calls, resending a dropped connection with the same request ids. With `spend`, its credits/calls are
    the image's shared totals and its request ids start at k0 * 100, so concurrent chains never share an id."""

    def __init__(self, witness, spend: "Spend | None" = None, k0: int = 0) -> None:
        super().__init__(witness)
        self._spend = spend
        self.k = k0

    @property
    def credits(self) -> float:  # type: ignore[override]
        return self._spend.credits if getattr(self, "_spend", None) else self.__dict__.get("_credits", 0.0)

    @credits.setter
    def credits(self, v: float) -> None:
        if getattr(self, "_spend", None):
            self._spend.credits = v
        else:
            self.__dict__["_credits"] = v

    @property
    def calls(self) -> int:  # type: ignore[override]
        return self._spend.calls if getattr(self, "_spend", None) else self.__dict__.get("_calls", 0)

    @calls.setter
    def calls(self, v: int) -> None:
        if getattr(self, "_spend", None):
            self._spend.calls = v
        else:
            self.__dict__["_calls"] = v

    async def ask(self, content, effort, model=pipeline.ASTRA, instructions=rewrite.P.SYSTEM, max_output=None):  # type: ignore[override]
        k = self.k
        for attempt in range(TRANSPORT_RETRIES + 1):
            self.k = k
            try:
                return await super().ask(content, effort, model, instructions, max_output)
            except openai.APIConnectionError as e:
                if attempt == TRANSPORT_RETRIES:
                    raise pipeline.ModelFailure(f"call {k}: transport failed {attempt + 1}x ({e})") from e
                logger.warning("call {}: transport failure ({}), resending in {}s", k, str(e)[:80], TRANSPORT_WAITS[attempt])
                await asyncio.sleep(TRANSPORT_WAITS[attempt])
        raise AssertionError("unreachable")


async def refine_with_sides(step: int, steps: int, photo: dict[str, str], grid: bytes, module: str,
                            witness: AsyncWitnessClient) -> list[dict[str, str]]:
    """sedge's refine turn + the SIDES sheet before the code block (the code stays last, as in sedge)."""
    content = refine_content(step, steps, photo, grid, module)
    sides, _ = await check(module, witness, SIDES)
    if sides is None:  # rendered fine from the front a moment ago: a renderer flake, so sedge's turn as it is
        logger.warning("sides render failed, refining from the front views only")
        return content
    return [*content[:-1], text_part(SIDES_LABEL), image_part(images.sheet_data_url(sides, SIDES.img_size)), content[-1]]


async def chain(calls: Calls, photo: dict[str, str], model: str, effort: str) -> tuple[str, bool]:
    """lumen.chain with the sides sheet in each rewrite."""
    best, grid = await first_module(calls, photo, model, effort)
    if grid is None:
        return best, False
    for step, (rewrite_model, rewrite_effort) in enumerate(lumen.REWRITES, 1):
        content = await refine_with_sides(step, len(lumen.REWRITES), photo, grid, best, calls.witness)
        try:
            text, incomplete = await calls.ask(content, rewrite_effort, rewrite_model)
        except pipeline.ModelFailure as e:
            logger.warning("rewrite {} call failed ({}), keeping the last accepted module", step, e)
            break
        if incomplete:
            logger.info("rewrite {} incomplete ({}), dropped", step, incomplete)
            continue
        cand = code_of(text)
        new_grid, errors = await check(cand, calls.witness, rewrite.VIEWS)
        if new_grid is None:
            logger.info("rewrite {} rejected ({}), dropped", step, errors[:1])
            continue
        logger.info("rewrite {} accepted", step)
        best, grid = cand, new_grid
    return best, True


# ---- v7: lineages to the end + a final judge-faithful duel ---------------------------------------------------------

# Each of sedge's two chains is a LINEAGE kept to the end: after its own draft, repairs and 2 rewrites it gets its own
# post refines (Sol high, with the sides sheet), instead of only the picked chain being refined.
LINEAGE_POST = ((pipeline.MODEL, 1), (pipeline.MODEL, 2), (pipeline.MODEL, 3))  # v8: 3 Sol-high post refines, each gated vs its parent
POST_STEPS = 3
# Fixed per-part budgets (sum = 5M), so the two concurrent lineages never see each other's progress: completion order
# cannot change which calls are made (replay-safe), and the image can never pass 5M credits / 60 calls.
LINEAGE_CREDITS, LINEAGE_CALLS = 2_250_000, 24  # v8: incl. 3 gates x 2 calls
DUEL_CREDITS_PER_CALL = 62_500  # 2 x 2.25M + 8 x 62.5K = 5.0M; 2 x 24 + 8 = 56 calls
LINEAGE_K0, DUEL_K0 = 40, 80  # request ids: lineage c from k = 40*c (ids 1.., 4001..), duel call i at k = 80+i (8001..)
# v8: output caps = reservations from measured maxima (loops 1-9: Astra high max 16,529 / Sol high max 19,716 output
# tokens, n = 657 / 1,645); about 30% headroom. The worst case reserved before each call shrinks, so more refines fit.
ASTRA_OUTPUT_CAP, SOL_OUTPUT_CAP = 22_000, 26_000
GATE_MARGIN = 0.5  # a refined child stays unless its parent scores better by more than this (pro-child: only clear regressions revert)
GATE_EFFORT, GATE_MAX_OUTPUT = "low", 2048

# The final pick copies the contest judge's stage 1 (404 judge-service multi_stage prompts, as in our local judge
# tools/s1.py): one call per front view and per presentation order, penalties 0-10, the lower total wins.
DUEL_MODEL, DUEL_EFFORT, DUEL_MAX_OUTPUT = pipeline.MODEL, "low", 2048
DUEL_VIEWS = RenderOptions(  # the judge's 4 front cameras at its resolution, white background, neutral light
    thetas=(30.0, 330.0, 0.0, 0.0), phis=(0.0, 0.0, 15.0, -30.0), img_size=1024, cam_radius=2.0,
    cam_fov_deg=49.1, gap=images.GRID_GAP, bg_color="ffffff", lighting="neutral",
)
DUEL_ANGLES = ("slightly left of the front", "slightly right of the front", "the front, from slightly below",
               "the front, from slightly above")
JUDGE_SYSTEM = ("You are a specialized 3D model evaluation system.\n"
                "Analyze visual quality and prompt adherence with expert precision.\n"
                "Always respond with valid JSON only.")
JUDGE_USER = ("You see two 3D models rendered from {angle_desc}.\n"
              "The reference image shows the target object.\n\n"
              "Which model is a more faithful 3D reproduction of the reference?\n\n"
              "Penalty 0-10:\n"
              "0 = Perfect match to reference\n"
              "3 = Minor issues (slight shape differences, missing small details)\n"
              "5 = Moderate issues (wrong style, significant details missing)\n"
              "7 = Major issues (wrong category but related, e.g. chair vs stool)\n"
              "10 = Completely wrong object\n\n"
              'Output: {{"penalty_1": <0-10>, "penalty_2": <0-10>, "issues": "<brief>"}}')


def budgeted(witness, k0: int, credits: float, calls: int) -> RCalls:
    """A Calls ledger whose guard sees only its own spend, inside a fixed share of the image's 5M / 60."""
    c = RCalls(witness, None, k0)
    c.credits, c.calls = float(pipeline.BUDGET_CREDITS - credits), pipeline.BUDGET_CALLS - calls
    return c


GATE_USER = ("You see two 3D models. Each image shows one model as a 2x2 sheet of renders from the front: top-left "
             "slightly left of the front, top-right slightly right of the front, bottom-left the front from slightly "
             "below, bottom-right the front from slightly above.\n"
             "The reference image shows the target object.\n\n"
             "Which model is a more faithful 3D reproduction of the reference?\n\n"
             "Penalty 0-10:\n0 = Perfect match to reference\n3 = Minor issues (slight shape differences, missing small details)\n"
             "5 = Moderate issues (wrong style, significant details missing)\n7 = Major issues (wrong category but related, e.g. chair vs stool)\n"
             "10 = Completely wrong object\n\n"
             'Output: {{"penalty_1": <0-10>, "penalty_2": <0-10>, "issues": "<brief>"}}')


async def gate(calls: Calls, photo: dict[str, str], parent_grid: bytes, child_grid: bytes, step: int) -> bool:
    """v8: keep the refined child unless its parent is clearly better. Two judge-rubric calls (both presentation orders)
    on the 4 front views of each, already rendered for the check. Unusable replies keep the child (as v7 did)."""
    sheets = (image_part(images.sheet_data_url(parent_grid, rewrite.VIEWS.img_size)),
              image_part(images.sheet_data_url(child_grid, rewrite.VIEWS.img_size)))
    totals, ok = [0.0, 0.0], 0
    for one, two in ((0, 1), (1, 0)):  # sequential: one ledger, fixed request-id order
        content = [text_part("Reference image (target object):"), photo, text_part("3D model 1:"), sheets[one],
                   text_part("3D model 2:"), sheets[two], text_part(GATE_USER)]
        try:
            text, _ = await calls.ask(content, GATE_EFFORT, pipeline.MODEL, instructions=JUDGE_SYSTEM,
                                      max_output=GATE_MAX_OUTPUT)
        except pipeline.ModelFailure as e:
            logger.warning("gate {} call failed ({}), keeping the child", step, e)
            return True
        p1, p2 = _pair_or_none(text)
        if p1 is None:
            continue
        totals[one] += p1
        totals[two] += p2
        ok += 1
    if not ok:
        return True
    parent, child = totals[0] / ok, totals[1] / ok
    keep = not (parent + GATE_MARGIN < child)
    logger.info("gate {}: parent {:.2f} child {:.2f} -> {}", step, parent, child, "child" if keep else "parent")
    return keep


def _pair_or_none(text: str) -> tuple[float | None, float | None]:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        d = json.loads(m.group(0)) if m else {}
        return (min(max(float(d["penalty_1"]), 0.0), 10.0), min(max(float(d["penalty_2"]), 0.0), 10.0))
    except (ValueError, KeyError, TypeError):
        return None, None


async def lineage(calls: Calls, photo: dict[str, str], model: str, effort: str) -> str | None:
    """One lineage: sedge's chain (draft, repairs, Sol-high -> Astra-high rewrites) + its own post refines, then the
    checked finish. -> the finished module, or None when no module was ever accepted."""
    best, ok = await chain(calls, photo, model, effort)
    if not ok:
        return None
    grid, _ = await check(best, calls.witness, rewrite.VIEWS)
    for post_model, step in LINEAGE_POST if grid is not None else ():
        content = await refine_with_sides(step, POST_STEPS, photo, grid, best, calls.witness)
        try:
            text, incomplete = await calls.ask(content, "high", post_model)
        except pipeline.ModelFailure as e:
            logger.warning("post refine {} call failed ({}), keeping the module so far", step, e)
            break
        if incomplete:
            continue
        cand = code_of(text)
        new_grid, errors = await check(cand, calls.witness, rewrite.VIEWS)
        if new_grid is None:
            logger.info("post refine {} rejected ({}), dropped", step, errors[:1])
            continue
        if not await gate(calls, photo, grid, new_grid, step):
            continue  # the parent was clearly better: keep it
        logger.info("post refine {} accepted", step)
        best, grid = cand, new_grid
    return await finish(best, calls.witness)


def _pair(text: str) -> tuple[float, float]:
    """A judge reply -> (penalty_1, penalty_2), clamped to 0-10; anything unusable -> (5, 5), which favours neither."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        d = json.loads(m.group(0)) if m else {}
        return (min(max(float(d["penalty_1"]), 0.0), 10.0), min(max(float(d["penalty_2"]), 0.0), 10.0))
    except (ValueError, KeyError, TypeError):
        return 5.0, 5.0


async def duel(witness, photo: dict[str, str], a: str, b: str) -> int:
    """Judge-faithful final pick between two finished modules -> 0 (a) or 1 (b). Both are rendered from the judge's 4
    front cameras at 1024 px; each view is judged in both presentation orders (8 calls); the lower summed penalty wins,
    a tie or a failed render keeps a (sedge's chain 1, its own tie rule)."""
    ga, _ = await check(a, witness, DUEL_VIEWS)
    gb, _ = await check(b, witness, DUEL_VIEWS)
    if ga is None or gb is None:
        logger.warning("duel: a final module did not render at judge resolution ({}), keeping lineage {}",
                       "a" if ga is None else "b", 1 if ga is None else 0)
        return 1 if ga is None and gb is not None else 0
    tiles = {0: images.view_data_urls(ga, DUEL_VIEWS.img_size), 1: images.view_data_urls(gb, DUEL_VIEWS.img_size)}
    jobs = []
    for order, (one, two) in enumerate(((0, 1), (1, 0))):
        for view, angle in enumerate(DUEL_ANGLES):
            content = [text_part("Reference image (target object):"), photo,
                       text_part("3D model 1:"), image_part(tiles[one][view]),
                       text_part("3D model 2:"), image_part(tiles[two][view]),
                       text_part(JUDGE_USER.format(angle_desc=angle))]
            calls = budgeted(witness, DUEL_K0 + 4 * order + view, DUEL_CREDITS_PER_CALL, 1)
            jobs.append((one, two, calls.ask(content, DUEL_EFFORT, DUEL_MODEL, instructions=JUDGE_SYSTEM,
                                             max_output=DUEL_MAX_OUTPUT)))
    replies = await asyncio.gather(*(job for _, _, job in jobs), return_exceptions=True)
    # v8: a view counts only when both presentation orders gave a usable reply (else it is left out)
    parsed = [(_pair_or_none(r[0]) if not isinstance(r, BaseException) else (None, None)) for r in replies]
    total, views = [0.0, 0.0], 0
    n = len(DUEL_ANGLES)
    for view in range(n):
        (a1, a2), (b1, b2) = parsed[view], parsed[n + view]  # order (0,1) and order (1,0)
        if a1 is None or b1 is None:
            continue
        total[0] += a1 + b2
        total[1] += a2 + b1
        views += 1
    if not views:
        logger.warning("duel: no view with both orders usable, keeping lineage 1")
        return 0
    winner = 1 if total[1] < total[0] else 0
    logger.info("duel ({} views): penalties lineage1 {} lineage2 {} -> lineage {}", views, total[0], total[1], winner + 1)
    return winner


async def generate(image: bytes, seed: int, witness: AsyncWitnessClient) -> str | None:
    """v7: two concurrent lineages (sedge's chains, each with its own post refines and finish), then a judge-faithful
    8-call duel picks one. Fixed per-part budgets keep it deterministic and within 5M credits / 60 calls."""
    pipeline.MAX_OUTPUT_TOKENS[pipeline.ASTRA] = ASTRA_OUTPUT_CAP
    pipeline.MAX_OUTPUT_TOKENS[pipeline.MODEL] = SOL_OUTPUT_CAP
    witness = CachedWitness(witness)
    photo = image_part(images.photo_data_url(image))
    finals = list(await asyncio.gather(*(
        lineage(budgeted(witness, LINEAGE_K0 * c, LINEAGE_CREDITS, LINEAGE_CALLS), photo, model, effort)
        for c, (model, effort) in enumerate(lumen.DRAFTS))))
    done = [m for m in finals if m]
    logger.info("lineages finished: {} of {}", len(done), len(finals))
    if not done:
        return None
    if len(done) == 1:
        return done[0]
    return done[await duel(witness, photo, done[0], done[1])]
