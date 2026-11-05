"""Qwen CQ backbone and image/video flow renderer."""

from torch import nn
import infra.parallel as P
from infra.cross_entropy import linear_cross_entropy
from models.legacy import RGBPatchLinear, FlowRenderer
from models.backbone_qwen35 import QwenBlock, RMSNorm
from models.cq import backbone
from models.generator_wan21 import Wan21Generator
from models.vision import QwenVisionInput


class OutputHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = P.make_embedding(cfg.vocab_size, cfg.width).weight
        # This untied parameter is used only by CE, so its gradient can go
        # straight into MCore's persistent FP32 accumulation buffer.
        self.weight._ce_direct_accumulation = True

    def forward(self, hidden, targets, chunk):
        return linear_cross_entropy(hidden, self.weight, targets, chunk)


class JointModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.validate_model(P.tp_size())
        self.cfg = cfg
        self.config = P._config
        self.embedding = P.make_embedding(cfg.vocab_size, cfg.width)
        if cfg.vision_encoder == "qwen35":
            self.patch_embed = QwenVisionInput(cfg)
        else:
            self.patch_embed = RGBPatchLinear(cfg.width, cfg.patch_size)
        self.layers = nn.ModuleList([QwenBlock(cfg, index) for index in range(cfg.layers)])
        self.final_norm = RMSNorm(cfg.width, cfg.norm_eps)
        if cfg.renderer == "wan":
            self.renderer = Wan21Generator(cfg)
        elif cfg.renderer == "moss":
            from models.moss import MOSSRenderer

            self.renderer = MOSSRenderer(cfg)
        else:
            self.renderer = FlowRenderer(cfg)
        self.lm_head = None if cfg.tie_word_embeddings else OutputHead(cfg)

    @property
    def text_query(self):
        # CPU / TP1 inspection convenience. Forward uses vocab-parallel lookup.
        return self.embedding.weight[self.cfg.text_mask_token_id]

    @property
    def image_query(self):
        return self.embedding.weight[self.cfg.image_mask_token_id]

    def backbone(self, batch, reference=False):
        return backbone(self, batch, reference)

    def forward(self, batch, reference=False, return_hidden=False, compute_image_loss=True):
        hidden = self.backbone(batch, reference=reference)
        # Match module/parameter registration order: renderer then untied LM
        # head. Backward can then reduce the LM-head bucket first and overlap
        # later buckets with renderer/backbone work, without queueing ready
        # renderer gradients behind an uncomputed output-head gradient.
        image_loss = None
        image_condition = None
        if batch.generation_image_count:
            indices = batch.mainline.condition_indices
            if batch.generation_image_indices is not None:
                indices = indices.index_select(0, batch.generation_image_indices)
            image_condition = hidden[indices.clamp_min(0)]
            if compute_image_loss:
                image_loss = self.renderer(image_condition, batch)
        text_hidden = hidden[batch.text_indices]
        targets = batch.ids.flatten()[batch.text_indices]
        if not targets.numel():
            text_loss = hidden.sum() * 0
        elif self.lm_head is not None:
            text_loss = self.lm_head(text_hidden, targets, self.cfg.ce_chunk)
        else:
            text_loss = linear_cross_entropy(text_hidden, self.embedding.weight, targets, self.cfg.ce_chunk)
        result = dict(text_loss=text_loss, image_loss=image_loss)
        if return_hidden:
            result["hidden"] = hidden
        if not compute_image_loss:
            result["image_conditions"] = image_condition
        return result
