"""Build an offline, audit-only image review gallery from existing JSONL files.

Thumbnails, styles, and filtering code are embedded. Original image links remain
relative local links, so retain the surrounding image directories when moving it.
No models are invoked and no review decisions or dataset records are changed.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
from html import escape
import io
import json
import os
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageOps


def text(value):
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def safe(value):
    return escape(text(value), quote=True)


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            rows.append(row)
    return rows


def index_rows(rows, label):
    indexed = {}
    for row in rows:
        sha = row.get("image_sha256")
        if (not isinstance(sha, str) or len(sha) != 64
                or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError(f"Invalid image_sha256 in {label}")
        if sha in indexed:
            raise ValueError(f"Duplicate image_sha256 in {label}: {sha}")
        indexed[sha] = row
    return indexed


def pretty(value):
    return escape(json.dumps(value, ensure_ascii=False, indent=2), quote=True)


def record_details(title, record, *, opened=False):
    opening = " open" if opened else ""
    return f"<details{opening}><summary>{safe(title)}</summary><pre>{pretty(record)}</pre></details>"


def disposition(record):
    if record is None:
        return "pending_final_review"
    for key in ("final_disposition", "disposition", "final_status", "status", "decision"):
        if record.get(key) is not None:
            return text(record[key])
    return "final_record_without_disposition"


def thumbnail(path):
    """Re-encode pixels, never embed original SVG/HTML or trust its MIME type."""
    with Image.open(path) as original:
        im = ImageOps.exif_transpose(original)
        im.thumbnail((560, 420), Image.Resampling.LANCZOS)
        if im.mode in {"RGBA", "LA"} or "transparency" in im.info:
            rgba = im.convert("RGBA")
            im = Image.new("RGB", rgba.size, "white")
            im.paste(rgba, mask=rgba.getchannel("A"))
        else:
            im = im.convert("RGB")
        buffer = io.BytesIO()
        im.save(buffer, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def render_review(model, review):
    if not isinstance(review, dict):
        return record_details(f"{model} · malformed review record", review)
    parsed = review.get("parsed")
    if not isinstance(parsed, dict):
        return record_details(f"{model} · missing / invalid judgment", review, opened=True)
    heading = " · ".join(text(x) for x in (
        model, parsed.get("assessment"), parsed.get("confidence"), parsed.get("image_form")))
    fields = [
        ("Description", parsed.get("image_description")),
        ("Reason", parsed.get("reason")),
        ("Needs provenance", parsed.get("needs_provenance")),
        ("Edit lineage", parsed.get("edit_lineage")),
    ]
    body = "".join(f"<dt>{safe(k)}</dt><dd>{safe(v)}</dd>" for k, v in fields)
    return (
        f"<details class='review' open><summary>{safe(heading)}</summary><dl>{body}</dl>"
        + record_details("Visible evidence", parsed.get("visible_evidence"), opened=True)
        + record_details("Counterevidence", parsed.get("counterevidence"), opened=True)
        + record_details("Complete model review record", review)
        + "</details>"
    )


def select_box(key, title, values):
    counts = Counter(values)
    options = "".join(
        f"<option value='{safe(value)}'>{safe(value)} ({counts[value]})</option>"
        for value in sorted(counts)
    )
    return (f"<label>{safe(title)}<select id='{key}'><option value=''>All</option>"
            f"{options}</select></label>")


STYLE = """
:root{color-scheme:light; font-family:system-ui,sans-serif;color:#182536;background:#eef2f6}
*{box-sizing:border-box}body{margin:0;padding:24px;max-width:1700px;margin-inline:auto}
h1{font-size:25px;margin:0 0 10px}h2{font-size:21px;margin:0 0 10px}p{line-height:1.5}
.notice{background:#fff6dc;border-left:4px solid #b68113;padding:12px 16px}
.filters{position:sticky;top:0;z-index:1;display:flex;flex-wrap:wrap;gap:12px;padding:12px;
background:#e5edf7;border:1px solid #c1d0e0;border-radius:10px;margin:18px 0}
label{display:flex;flex-direction:column;gap:5px;font-size:13px;font-weight:600;min-width:180px}
select,input,button{font:inherit;border:1px solid #abb8c7;border-radius:5px;padding:8px;background:white}
button{align-self:flex-end;cursor:pointer}#count{align-self:flex-end;padding:9px 0;margin:0}
.card{background:white;border:1px solid #d4dde6;border-radius:12px;margin:18px 0;padding:18px}
.card[hidden]{display:none}.columns{display:grid;grid-template-columns:minmax(280px,560px) 1fr;gap:20px}
.picture{text-align:center;background:#f7f8fa;border:1px solid #e1e5e9;border-radius:8px;padding:8px}
.picture img{max-width:100%;height:auto;max-height:420px;object-fit:contain}
.image-error{color:#a12727}a{color:#185a9d}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;
background:#f5f7fa;padding:10px;border-radius:5px;max-height:420px;overflow:auto}
.badge{display:inline-block;font-size:12px;background:#e7eef8;padding:5px 8px;border-radius:4px;
margin:0 5px 7px 0;overflow-wrap:anywhere}details{margin:9px 0;border-top:1px solid #e4e9ef;padding-top:9px}
summary{font-weight:600;cursor:pointer;overflow-wrap:anywhere}dl{margin:8px 0;font-size:14px}
dt{font-weight:600;margin-top:7px}dd{margin:3px 0 0;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere}
.metadata{font-size:12px;color:#536478;overflow-wrap:anywhere}.review{padding:12px;border:1px solid #dce4ed;
border-radius:7px}.empty{padding:30px;color:#52616e}.small{font-size:13px}
@media(max-width:900px){body{padding:12px}.columns{grid-template-columns:1fr}.filters{position:static}}
"""

SCRIPT = """
const cards=Array.from(document.querySelectorAll('.card'));
const group=document.getElementById('group'), state=document.getElementById('state');
const domain=document.getElementById('domain'), search=document.getElementById('search');
function filter(){
  const query=search.value.trim().toLowerCase();let visible=0;
  cards.forEach(card=>{
    const match=(!group.value||card.dataset.group===group.value)&&
      (!state.value||card.dataset.state===state.value)&&
      (!domain.value||card.dataset.domain===domain.value)&&
      (!query||card.dataset.search.includes(query));
    card.hidden=!match;if(match)visible++;
  });
  document.getElementById('count').textContent=visible+' / '+cards.length+' images';
  document.getElementById('empty').hidden=visible!==0;
}
[group,state,domain,search].forEach(input=>input.addEventListener('input',filter));
document.getElementById('reset').addEventListener('click',()=>{
  [group,state,domain,search].forEach(input=>input.value='');filter();
});filter();
"""


def build_gallery(samples_path, blind_path, final_path, output_path):
    samples_path, output_path = Path(samples_path).resolve(), Path(output_path).resolve()
    inputs = [samples_path, Path(blind_path).resolve()]
    if final_path:
        inputs.append(Path(final_path).resolve())
    if output_path in inputs:
        raise ValueError("Output must not overwrite an input")
    samples = index_rows(load_jsonl(samples_path), "samples")
    if not samples:
        raise ValueError("Sample manifest is empty")
    blind = index_rows(load_jsonl(blind_path), "blind decisions")
    final = index_rows(load_jsonl(final_path), "final decisions") if final_path else {}
    unknown = (set(blind) | set(final)) - set(samples)
    if unknown:
        raise ValueError(f"Decision files contain {len(unknown)} images absent from samples")
    cards, groups, states, domains = [], [], [], []
    image_errors = []
    for sha, sample in samples.items():
        row = blind.get(sha)
        final_row = final.get(sha)
        review_id = sample.get("review_id") or (row or {}).get("review_id") or sha[:12]
        group = text(sample.get("pilot_group") or "unspecified")
        domain = text(sample.get("domain") or "unspecified")
        state = disposition(final_row)
        groups.append(group)
        domains.append(domain)
        states.append(state)
        path = Path(sample["image_path"]) if sample.get("image_path") else None
        if path is not None and not path.is_absolute():
            path = samples_path.parent / path
        try:
            if path is None:
                raise ValueError("Sample has no image_path")
            thumb = thumbnail(path)
            picture = f"<img src='{thumb}' alt='Review image {safe(review_id)}' loading='lazy'>"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            image_errors.append({"review_id": review_id, "error": error})
            picture = f"<p class='image-error'>Thumbnail unavailable: {safe(error)}</p>"
        if path is not None:
            relative = quote(os.path.relpath(path.resolve(), output_path.parent), safe="/")
            picture += f"<p><a href='{safe(relative)}' target='_blank' rel='noopener'>Open original image</a></p>"
        source_fields = {k: v for k, v in sample.items()
                         if k.startswith("source") or k.startswith("ground_truth")
                         or k in {"associated_sources", "caption_origin", "authenticity_ground_truth",
                                  "ai_edited_from_real_ground_truth", "generation_metadata", "domain_label_scope"}}
        scores = sample.get("detector_scores", sample.get("scores", {}))
        summary = (
            f"<h2>{safe(review_id)} · {safe(sample.get('source'))}</h2>"
            f"<span class='badge'>{safe(group)}</span><span class='badge'>{safe(domain)}</span>"
            f"<span class='badge'>Final: {safe(state)}</span>"
            f"<span class='badge'>Blind: {safe((row or {}).get('status', 'missing_blind_decision'))}</span>"
            f"<p class='metadata'>SHA256: {safe(sha)}</p>"
        )
        reviews = (row or {}).get("reviews", {})
        if not isinstance(reviews, dict):
            reviews_html = record_details("Malformed blind reviews", reviews, opened=True)
        else:
            reviews_html = "".join(render_review(model, review) for model, review in reviews.items())
        if row is None:
            reviews_html = "<p>Blind decision is not available yet.</p>"
        final_html = (record_details("Final record and adjudication", final_row, opened=True)
                      if final_row is not None else "<p class='small'>Final disposition / adjudication is pending.</p>")
        search_text = " ".join((text(review_id), sha, text(sample.get("source")))).lower()
        cards.append(
            f"<article class='card' data-group='{safe(group)}' data-state='{safe(state)}' "
            f"data-domain='{safe(domain)}' data-search='{safe(search_text)}'>"
            + summary + "<div class='columns'><div><div class='picture'>" + picture + "</div>"
            + record_details("Original detector scores", scores, opened=True)
            + record_details("Registered source evidence / labels", source_fields, opened=True)
            + record_details("Complete original sample record", sample)
            + "</div><div>" + final_html + reviews_html
            + record_details("Complete blind decision", row) + "</div></div></article>"
        )
    count = len(samples)
    filters = (select_box("group", "Pilot group", groups)
               + select_box("state", "Final disposition", states)
               + select_box("domain", "Domain", domains)
               + "<label>Review ID / source / SHA<input id='search' type='search' placeholder='A001'></label>"
               + "<button id='reset' type='button'>Reset</button><p id='count'></p>")
    provenance = {"samples": str(samples_path), "blind_decisions": str(Path(blind_path).resolve()),
                  "final_decisions": str(Path(final_path).resolve()) if final_path else None,
                  "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "sample_count": count, "missing_blind_count": count - len(blind),
                  "missing_final_count": count - len(final), "thumbnail_errors": image_errors}
    document = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>DCVLM API review gallery</title><style>" + STYLE + "</style></head><body>"
        "<h1>DCVLM API review gallery</h1>"
        "<p class='notice'>Audit only. Detector scores and visual-model agreement do not establish image "
        "provenance. This gallery does not change training eligibility or delete any data. "
        "Thumbnails display the original image, not the detector crop.</p>"
        + record_details("Inputs and build status", provenance)
        + "<div class='filters'>" + filters + "</div>"
        + "<p class='empty' id='empty' hidden>No images match these filters.</p>"
        + "".join(cards) + "<script>" + SCRIPT + "</script></body></html>"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output_path)
    return {**provenance, "output": str(output_path), "html_bytes": output_path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, help="Pilot sample manifest JSONL")
    parser.add_argument("--blind-decisions", required=True, help="aggregate.py decisions.jsonl")
    parser.add_argument("--final-decisions", help="Optional final JSONL, keyed by image_sha256")
    parser.add_argument("--output", required=True, help="Output HTML path")
    args = parser.parse_args()
    print(json.dumps(build_gallery(args.samples, args.blind_decisions, args.final_decisions, args.output),
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
