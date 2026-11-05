"""Same-position CQ text decoding."""
import math

import torch
from torch.nn import functional as F


def _sample(logits, temperature, top_k, top_p):
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits.float() / temperature
    if top_k:
        threshold = logits.topk(min(top_k, logits.shape[-1]), -1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    if top_p < 1:
        sorted_logits, order = logits.sort(descending=True)
        remove = sorted_logits.softmax(-1).cumsum(-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        logits = logits.masked_fill(torch.zeros_like(remove).scatter(1, order, remove), -torch.inf)
    return torch.multinomial(F.softmax(logits, -1), 1).squeeze(-1)



@torch.no_grad()
def generate_text(model, batch, *, max_new_tokens=128, temperature=0.0, top_k=0,
                  top_p=1.0, eos_token_id=None, use_cache=False, return_logits=False):
    """Same-position CQ decoding. Qwen3.5 recomputes the full prefix each step."""
    if use_cache:
        raise NotImplementedError("Qwen3.5 CQ generation currently supports no-cache decoding only")
    if max_new_tokens < 0 or top_k < 0 or not 0 < top_p <= 1 or not math.isfinite(float(temperature)):
        raise ValueError("Invalid text generation settings")
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        prepared = model.prepare_inputs(batch)
        valid = prepared["token_types"].ne(3) & prepared["segment_ids"].ge(0)
        if bool((~valid.any(-1)).any()):
            raise ValueError("Each prompt needs nonempty content")
        for row in range(valid.shape[0]):
            if prepared["segment_ids"][row, valid[row]].unique().numel() != 1:
                raise ValueError("Unpack independent segments before generation")
        sequences = [row[keep].clone() for row, keep in zip(prepared["input_ids"], valid)]
        device = prepared["input_ids"].device
        text_config = model.config.qwen_config.text_config
        eos = text_config.eos_token_id if eos_token_id is None else eos_token_id
        eos_ids = [] if eos is None else (list(eos) if isinstance(eos, (list, tuple)) else [eos])
        pad = text_config.pad_token_id
        pad = 0 if pad is None else pad
        # Frozen input features are reusable inputs, not decoder KV/GDN caches.
        observed = model.backbone._video_flags(batch, "video_observed_mask",
                                              len(prepared["video_token_indices"]), device, True)
        features = [prepared["inputs_embeds"][row, slots] if observed[i] else None
                    for i, (row, slots) in enumerate(zip(prepared["video_batch_indices"],
                                                        prepared["video_token_indices"]))]
        fixed = dict(video_grid_thw=prepared["video_grid_thw"], video_features=features,
                     video_observed_mask=observed, target_video_ids=torch.empty(0, device=device, dtype=torch.long))
        generated, scores = [], []
        finished = torch.zeros(len(sequences), device=device, dtype=torch.bool)
        for _ in range(max_new_tokens):
            lengths = torch.tensor([s.numel() for s in sequences], device=device)
            ids = prepared["input_ids"].new_full((len(sequences), int(lengths.max()) + 1), pad)
            attention_mask = torch.zeros_like(ids, dtype=torch.bool)
            for row, seq in enumerate(sequences):
                ids[row, :seq.numel()] = seq
                attention_mask[row, :seq.numel()] = True
                ids[row, seq.numel()] = model.config.mask_token_id
                attention_mask[row, seq.numel()] = ~finished[row]
            step = model.prepare_inputs({**fixed, "input_ids": ids, "attention_mask": attention_mask})
            hidden = model.backbone(step)["query_hidden"]
            logits = model.lm_head(hidden[torch.arange(len(sequences), device=device), lengths])
            sampling_logits = logits.clone()
            # Input placeholders are not output text. Keep raw logits for parity checks.
            reserved = {model.config.mask_token_id, model.config.video_mask_token_id,
                        model.config.qwen_config.video_token_id, model.config.qwen_config.image_token_id}
            sampling_logits[:, list(reserved)] = -torch.inf
            token = _sample(sampling_logits, temperature, top_k, top_p)
            token = torch.where(finished, pad, token)
            generated.append(token)
            if return_logits:
                scores.append(logits)
            for row in range(len(sequences)):
                if not finished[row]:
                    sequences[row] = torch.cat((sequences[row], token[row:row + 1]))
            for eos_id in eos_ids:
                finished |= token.eq(eos_id)
            if bool(finished.all()):
                break
        tokens = torch.stack(generated, 1) if generated else prepared["input_ids"].new_empty((len(sequences), 0))
        if return_logits:
            logits = torch.stack(scores, 1) if scores else model.lm_head.weight.new_empty((len(sequences), 0, text_config.vocab_size))
            return {"sequences": tokens, "logits": logits}
        return tokens
    finally:
        for module, mode in modes:
            module.training = mode
