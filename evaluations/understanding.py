"""Same-position CQ text likelihood and image-to-text understanding metrics.

Text benchmark choices use original-choice character normalization; I2T uses
mean token log likelihood and the source fixed-alpha language-prior correction.
No external language or contrastive model scores UMM predictions.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F

from evaluations.validation import evaluation_state


@torch.no_grad()
def score_batch(model, batch):
    """Return per-row total/mean target log likelihood without causal shifting."""
    labels = batch["labels"]
    valid = labels.ne(-100)
    if "token_types" in batch:
        valid &= batch["token_types"].ne(1)
    output = model(batch)
    logits, targets = output["text_logits"], output["text_labels"]
    if logits.ndim == 3:
        logits, targets = logits[valid], labels[valid]
    if targets is None or not torch.equal(targets, labels[valid]):
        raise ValueError("model labels do not match same-position target mask")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("nonfinite text logits")
    logp = -F.cross_entropy(logits.float(), targets, reduction="none")
    rows = valid.nonzero(as_tuple=True)[0]
    sums = torch.zeros(labels.shape[0], device=logp.device, dtype=logp.dtype)
    sums.index_add_(0, rows, logp)
    counts = valid.sum(dim=1)
    if bool((counts <= 0).any()):
        raise ValueError("every scored sequence must have target tokens")
    errors = torch.zeros(labels.shape[0], device=logp.device, dtype=torch.long)
    errors.index_add_(0, rows, logits.argmax(-1).ne(targets).long())
    return {"loglikelihood": sums, "mean_loglikelihood": sums / counts,
            "token_count": counts, "greedy": errors.eq(0)}


def _gather_rows(rows, distributed):
    if distributed and dist.is_available() and dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, rows)
        return [row for part in gathered for row in part]
    return rows


def evaluate_perplexity(model, batches, *, ema=None, distributed=True):
    total_logp, total_tokens, sequences = 0.0, 0, 0
    with evaluation_state(model, ema=ema):
        for batch in batches:
            scored = score_batch(model, batch)
            total_logp += float(scored["loglikelihood"].sum())
            total_tokens += int(scored["token_count"].sum())
            sequences += len(scored["token_count"])
    totals = _gather_rows([(total_logp, total_tokens, sequences)], distributed)
    logp, tokens, count = map(sum, zip(*totals))
    if tokens <= 0:
        raise ValueError("cannot score empty text evaluation")
    nll = -logp / tokens
    return {"loss": nll, "perplexity": math.exp(min(nll, 100)),
            "target_tokens": tokens, "sequences": count, "scoring": "same_position_query"}


def evaluate_choices(model, batches, answers, *, ema=None, distributed=True):
    """Prepared batches include question_ids, choice_ids, normalization_chars.

    ``answers`` maps question IDs to correct zero-based choices. Scores retain
    both sum and original-character-normalized likelihood; no token-length
    approximation replaces the source text benchmark contract.
    """
    rows = []
    with evaluation_state(model, ema=ema):
        for batch in batches:
            scored = score_batch(model, batch)
            metadata = [batch[key] for key in ("question_ids", "choice_ids", "normalization_chars")]
            if any(len(items) != len(scored["loglikelihood"]) for items in metadata):
                raise ValueError("choice metadata does not align with sequence rows")
            for index, (question, choice, chars) in enumerate(zip(*metadata)):
                if int(chars) <= 0:
                    raise ValueError("choice character normalization must be positive")
                value = float(scored["loglikelihood"][index])
                rows.append((str(question), int(choice), value, value / int(chars)))
    groups = {}
    for question, choice, raw, normalized in _gather_rows(rows, distributed):
        group = groups.setdefault(question, {})
        if choice in group:
            raise ValueError("duplicate question/choice in evaluation")
        group[choice] = (raw, normalized)
    answers = {str(key): int(value) for key, value in answers.items()}
    if not groups or set(groups) != set(answers):
        raise ValueError("choice evaluation coverage differs from answer identities")
    correct, normalized_correct = 0, 0
    for question, choices in groups.items():
        if set(choices) != set(range(len(choices))) or len(choices) < 2 or answers[question] not in choices:
            raise ValueError("choice IDs must cover complete consecutive candidate sets")
        ordered = sorted(choices)
        # Ascending candidate order preserves source argmax tie breaking.
        correct += max(ordered, key=lambda i: choices[i][0]) == answers[question]
        normalized_correct += max(ordered, key=lambda i: choices[i][1]) == answers[question]
    return {"accuracy": correct / len(groups), "accuracy_normalized": normalized_correct / len(groups),
            "questions": len(groups), "normalization": "original_choice_characters"}


def language_prior_debiased_scores(scores, *, alpha=1.0):
    """Source protocol: mean-token log P(text|image) minus image logmeanexp."""
    if scores.ndim != 2 or min(scores.shape) <= 0 or not bool(torch.isfinite(scores).all()):
        raise ValueError("conditional scores must be a complete finite image-text matrix")
    if not math.isfinite(float(alpha)) or float(alpha) != 1.0:
        raise ValueError("formal language-prior correction fixes alpha=1")
    scores = scores.float()
    prior = torch.logsumexp(scores, dim=0) - math.log(scores.shape[0])
    return scores - prior.unsqueeze(0), prior


def score_image_text_candidates(model, batches, *, image_count, text_count, ema=None, distributed=True):
    """Each prepared row pairs one known image with one target candidate text.

    Batches include image_indices/text_indices identifying the full score matrix.
    Strict coverage is required before language-prior calibration.
    """
    rows = []
    with evaluation_state(model, ema=ema):
        for batch in batches:
            scores = score_batch(model, batch)["mean_loglikelihood"]
            if len(batch["image_indices"]) != len(scores) or len(batch["text_indices"]) != len(scores):
                raise ValueError("candidate metadata length mismatch")
            rows.extend((int(i), int(t), float(value)) for i, t, value in
                        zip(batch["image_indices"], batch["text_indices"], scores))
    matrix = torch.full((image_count, text_count), torch.nan)
    for image, text, score in _gather_rows(rows, distributed):
        if not (0 <= image < image_count and 0 <= text < text_count) or not bool(torch.isnan(matrix[image, text])):
            raise ValueError("duplicate or out-of-range image-text candidate")
        matrix[image, text] = score
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("image-text scoring did not cover the full candidate matrix")
    return matrix


def classification_metrics(scores, labels, *, calibrated=True):
    if calibrated:
        scores, _ = language_prior_debiased_scores(scores)
    labels = torch.as_tensor(labels, device=scores.device, dtype=torch.long)
    if labels.shape != (scores.shape[0],) or bool(((labels < 0) | (labels >= scores.shape[1])).any()):
        raise ValueError("classification labels do not match candidate scores")
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    return {"top1": float(order[:, 0].eq(labels).float().mean()),
            "top5": float(order[:, :min(5, scores.shape[1])].eq(labels[:, None]).any(1).float().mean()),
            "samples": len(labels), "language_prior_alpha": 1.0 if calibrated else 0.0}


def retrieval_metrics(scores, image_ids, text_image_ids, *, calibrated=True, ks=(1, 5, 10)):
    if calibrated:
        scores, _ = language_prior_debiased_scores(scores)
    if scores.shape != (len(image_ids), len(text_image_ids)):
        raise ValueError("retrieval identities do not match score matrix")
    result = {}
    for name, matrix, queries, candidates in (
        ("image_to_text", scores, image_ids, text_image_ids),
        ("text_to_image", scores.T, text_image_ids, image_ids),
    ):
        relevant = torch.tensor([[q == c for c in candidates] for q in queries], device=matrix.device)
        if not bool(relevant.any(1).all()):
            raise ValueError("every retrieval query requires a matching candidate")
        order = matrix.argsort(dim=1, descending=True, stable=True)
        for k in ks:
            if k <= 0:
                raise ValueError("retrieval K must be positive")
            result[f"{name}/recall@{k}"] = float(relevant.gather(1, order[:, :k]).any(1).float().mean())
    return result
