"""Combine unweighted model objectives using trainer-owned weights and DP counts."""


def compose_losses(outputs, cfg, scales=None):
    text_scale, image_scale = scales if scales is not None else (1.0, 1.0)
    loss = outputs["text_loss"] * (cfg.text_weight * text_scale)
    image = outputs["image_loss"]
    if image is not None:
        loss = loss + image * (cfg.image_weight * image_scale)
    else:
        image = outputs["text_loss"].detach().new_zeros(())
    return dict(outputs, loss=loss, image_loss=image)
