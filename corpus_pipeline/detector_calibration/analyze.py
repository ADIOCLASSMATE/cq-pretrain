"""Analyze a locked DCVLM detector sample; scores are not provenance labels.

No positive provenance labels means no precision/recall/AUC claims. Calibration
and holdout membership must have been assigned before inference. Contact sheets
are review aids, never an automatically generated ground-truth annotation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np


MODELS = ("cf224", "cf384", "unifd")
THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
KEEP_ORIGINS = {"natural", "programmatic", "rendered"}


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def wilson(k, n, z=1.959963984540054):
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - radius), min(1.0, center + radius)]


def domain(row):
    return row.get("source_domain", row.get("domain", "unknown"))


def origin(row):
    return row.get("image_origin", row.get("source_image_origin", "unknown"))


def split(row):
    value = row.get("split", row.get("calibration_split"))
    if value not in {"calibration", "holdout"}:
        raise ValueError("Missing preassigned calibration/holdout split")
    return value


def valid_prob(row, model):
    score = row.get("scores", {}).get(model, {})
    prob = score.get("prob")
    if prob is None or not math.isfinite(float(prob)):
        return None
    if not 0 <= float(prob) <= 1:
        raise ValueError(f"Invalid probability for {model}")
    return float(prob)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    return {
        "n": n,
        "quantiles": {str(q): float(np.quantile(values, q)) for q in (0, .5, .9, .95, .99, 1)} if n else {},
        "thresholds": {
            str(t): {"flagged": int((values >= t).sum()),
                     "fraction": float((values >= t).mean()) if n else None,
                     "wilson95_descriptive": wilson(int((values >= t).sum()), n)}
            for t in THRESHOLDS
        },
    }


def empirical_threshold(values, target):
    """Smallest float threshold with calibration exceedance <= target.

    Values above 1 explicitly mean disabled. This controls an empirical rate in
    source-supported negatives, not precision or a population guarantee.
    """
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    candidates = [0.0, *np.nextafter(np.unique(values), math.inf)]
    return next(float(t) for t in candidates if float((values >= t).mean()) <= target)


def analyze(samples, scores):
    by_id = {row["image_sha256"]: row for row in samples}
    if len(by_id) != len(samples):
        raise ValueError("Sample image hashes must be unique")
    merged = []
    for record in scores:
        item = dict(by_id[record["image_sha256"]])
        item["scores"] = record.get("scores", {})
        split(item)
        merged.append(item)
    if len({r["image_sha256"] for r in merged}) != len(merged):
        raise ValueError("Duplicate prediction rows")
    result = {
        "scope": "source_stratified_diagnostic_sample_not_population",
        "source_labels_are_not_pixel_verified_ground_truth": True,
        "sample_count": len(samples), "score_record_count": len(scores),
        "missing_score_records": len(samples) - len(scores),
        "domains": dict(Counter(domain(r) for r in samples)),
        "source_origins": dict(Counter(origin(r) for r in samples)),
        "split_counts": dict(Counter(split(r) for r in samples)),
        "precision_recall_auc": None,
        "positive_label_limitation": "No positive ground truth is inferred from model scores or visual appearance.",
        "interval_limitation": "Wilson intervals describe sampled images; source clustering/unequal sampling prevent corpus-wide coverage claims.",
        "models": {}, "pairwise": {},
    }
    for model in MODELS:
        groups = defaultdict(list)
        supported = defaultdict(list)
        valid = []
        for r in merged:
            prob = valid_prob(r, model)
            if prob is None:
                continue
            valid.append(prob)
            groups["domain/" + domain(r)].append(prob)
            groups["origin/" + origin(r)].append(prob)
            groups["split/" + split(r)].append(prob)
            w, h = r.get("decoded_width", 0), r.get("decoded_height", 0)
            if w and h:
                groups["input_size/" + ("min_lt224" if min(w, h) < 224 else "min_ge224")].append(prob)
                groups["aspect/" + ("ge3" if max(w, h) / min(w, h) >= 3 else "lt3")].append(prob)
            if origin(r) in KEEP_ORIGINS:
                supported[split(r)].append(prob)
                groups["source_supported_keep/" + split(r)].append(prob)
        candidates = {}
        for target in (.05, .01, .005):
            t = empirical_threshold(supported["calibration"], target)
            held = np.asarray(supported["holdout"])
            k = int((held >= t).sum()) if t is not None else None
            candidates[str(target)] = {
                "threshold": t, "disabled_above_one": t is not None and t > 1,
                "calibration_source_supported_keep_n": len(supported["calibration"]),
                "holdout_source_supported_keep_n": len(held),
                "holdout_flagged": k,
                "holdout_fraction": k / len(held) if k is not None and len(held) else None,
                "holdout_wilson95_descriptive": wilson(k, len(held)) if k is not None else None,
                "use": "negative-control review-threshold candidate; not automatic deletion threshold",
            }
        result["models"][model] = {"all": stats(valid), "groups": {k: stats(v) for k, v in groups.items()},
                                   "negative_control_calibration": candidates}
    for a, b in (("cf224", "cf384"), ("cf384", "unifd"), ("cf224", "unifd")):
        pairs = [(valid_prob(r, a), valid_prob(r, b)) for r in merged
                 if valid_prob(r, a) is not None and valid_prob(r, b) is not None]
        result["pairwise"][a + "/" + b] = {
            "n": len(pairs), "thresholds": {
                str(t): {"both_flagged": sum(x >= t and y >= t for x, y in pairs),
                         "either_flagged": sum(x >= t or y >= t for x, y in pairs),
                         "disagree": sum((x >= t) != (y >= t) for x, y in pairs)} for t in THRESHOLDS}}
    return result, merged


def review_candidates(rows, seed=20261022):
    selected, reasons = {}, defaultdict(list)

    def add(items, why):
        for r in items:
            key = r["image_sha256"]
            selected[key] = r
            if why not in reasons[key]:
                reasons[key].append(why)

    for model in MODELS:
        valid = [r for r in rows if valid_prob(r, model) is not None]
        val = lambda r: valid_prob(r, model)
        add(sorted(valid, key=val, reverse=True)[:12], "high_score/" + model)
        for t in (.5, .9, .99):
            add(sorted(valid, key=lambda r: abs(val(r) - t))[:2], f"boundary/{model}/{t}")
    valid = [r for r in rows if all(valid_prob(r, m) is not None for m in ("cf384", "unifd"))]
    for sign in (1, -1):
        add(sorted(valid, key=lambda r: sign * (r["scores"]["cf384"]["prob"] - r["scores"]["unifd"]["prob"]), reverse=True)[:8],
            "disagreement/" + str(sign))
    rng = random.Random(seed)
    for group in sorted({domain(r) for r in rows}):
        candidates = sorted([r for r in rows if domain(r) == group], key=lambda r: r["image_sha256"])
        rng.shuffle(candidates)
        add(candidates[:4], "domain_random/" + group)
    add(sorted(valid, key=lambda r: max(r["scores"][m]["prob"] for m in ("cf384", "unifd")))[:6], "low_score_control")
    result = []
    for i, (key, row) in enumerate(selected.items(), 1):
        result.append({**row, "review_id": f"R{i:03}", "selection_reasons": reasons[key],
                       "review_status": "pending_visual_inspection"})
    return result


def contact_sheets(rows, out):
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    out = Path(out); out.mkdir(exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    paths = []
    for start in range(0, len(rows), 8):
        page = rows[start:start + 8]
        canvas = Image.new("RGB", (1400, 1040), "#dddddd")
        draw = ImageDraw.Draw(canvas)
        for pos, r in enumerate(page):
            x, y = (pos % 4) * 350, (pos // 4) * 520
            with Image.open(r["image_path"]) as raw:
                img = ImageOps.exif_transpose(raw).convert("RGB")
                img.thumbnail((340, 475))
                canvas.paste(img, (x + (350 - img.width) // 2, y + 30 + (475 - img.height) // 2))
            # Scores and source labels deliberately omitted during visual review.
            draw.text((x + 8, y + 5), r["review_id"], font=font, fill="black")
        path = out / f"page_{start // 8 + 1:02}.jpg"
        canvas.save(path, quality=94)
        paths.append(str(path.resolve()))
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--review-split", choices=("calibration", "holdout"), default="calibration",
                   help="Inspect calibration to select thresholds; inspect holdout only after freezing them.")
    args = p.parse_args(); args.output.mkdir(exist_ok=True, parents=True)
    summary, rows = analyze(read_jsonl(args.samples), read_jsonl(args.scores))
    summary["inputs"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.samples, args.scores)}
    write_json(args.output / "analysis.json", summary)
    review = review_candidates([r for r in rows if split(r) == args.review_split])
    (args.output / "review_candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review))
    sheets = contact_sheets(review, args.output / "contact_sheets")
    write_json(args.output / "review_pages.json", {"review_count": len(review), "review_split": args.review_split, "sheets": sheets})
    print(json.dumps({"sample_count": len(rows), "review_count": len(review), "pages": len(sheets)}))


if __name__ == "__main__":
    main()
