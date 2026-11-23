"""Explicit image-target permissions; understanding inputs are never removed."""

CONTRACT = "generation_target_occurrence_v1"
FIELDS = ("image_loss_allowed", "dcvlm_image_loss_allowed")
TARGET_FIELDS = (*FIELDS, "generation_target")


def image_loss_allowed(*owners):
    """Check the published quality and usage vetoes for the selected media."""
    flags = bound_flags(*owners)
    return all(flags.get(key, True) for key in FIELDS)


def generation_target(*owners):
    """Only the occurrence can grant loss. Every owner can veto it."""
    flags = bound_flags(*owners)
    return owners[-1].get("generation_target") is True and all(flags.values())


def bound_flags(*owners):
    """Keep sparse grants and vetoes separate when a release binds original media."""
    flags = {}
    for owner in owners:
        for key in TARGET_FIELDS:
            if key in owner:
                if type(owner[key]) is not bool:
                    raise ValueError(f"{key} must be a boolean when present")
                flags[key] = flags.get(key, True) and owner[key]
    return flags


def occurrence_images(row, span):
    """Logical asset and the actual bytes selected for this occurrence."""
    index = span["image_index"]
    if type(index) is not int or not 0 <= index < len(row["images"]):
        raise ValueError("Image occurrence has an out-of-range image_index")
    image = row["images"][index]
    return image, span.get("upstream_image") or image


def occurrence_flags(*owners):
    return dict(
        image_loss_allowed=image_loss_allowed(*owners),
        generation_target=generation_target(*owners),
        non_cfg=next((o["non_cfg"] for o in reversed(owners) if o.get("non_cfg") is not None), None),
    )
