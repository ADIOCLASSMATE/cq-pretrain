"""Explicit image-target permissions; understanding inputs are never removed."""

CONTRACT = 'image_loss_boolean_and_v1'
FIELDS = ('image_loss_allowed', 'dcvlm_image_loss_allowed')


def image_loss_allowed(*owners):
    """Combine record/image flags without permitting a narrower override.

    Missing fields preserve historical training behavior. Present fields must
    be actual booleans: nulls, numbers and strings cannot silently enable or
    disable an objective. Validate every field even after a denial is found.
    """
    allowed = True
    for owner in owners:
        for name in FIELDS:
            if name in owner:
                value = owner[name]
                if type(value) is not bool:
                    raise ValueError(f'{name} must be a boolean when present')
                allowed = allowed and value
    return allowed
