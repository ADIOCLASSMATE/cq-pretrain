"""Explicit image-target permissions; understanding inputs are never removed."""

CONTRACT = 'image_loss_boolean_and_v1'
FIELDS = ('image_loss_allowed', 'dcvlm_image_loss_allowed')
TARGET_FIELDS = (*FIELDS, 'generation_target')


def image_loss_allowed(*owners, grant_field=None):
    """Combine record/image flags without permitting a narrower override.

    A native Wan grant belongs to the occurrence. A release may explicitly
    bind its published image_loss_allowed masks instead. Every veto still applies.
    Without a grant field this checks the two legacy quality/usage restrictions.
    """
    fields = TARGET_FIELDS if grant_field else FIELDS
    allowed = True
    for owner in owners:
        for name in fields:
            if name in owner:
                value = owner[name]
                if type(value) is not bool:
                    raise ValueError(f'{name} must be a boolean when present')
                allowed = allowed and value
    if grant_field == 'generation_target':
        return allowed and owners[-1].get('generation_target') is True
    if grant_field:
        return allowed and any(o.get(grant_field) is True for o in owners)
    return allowed


def bound_flags(*owners):
    """Keep sparse grants and vetoes separate when a release binds original media."""
    flags = {}
    for owner in owners:
        for key in TARGET_FIELDS:
            if key in owner:
                if type(owner[key]) is not bool:
                    raise ValueError(f'{key} must be a boolean when present')
                flags[key] = flags.get(key, True) and owner[key]
    return flags


def occurrence_images(row, span):
    """Logical asset and the actual bytes selected for this occurrence."""
    index = span['image_index']
    if type(index) is not int or not 0 <= index < len(row['images']):
        raise ValueError('Image occurrence has an out-of-range image_index')
    image = row['images'][index]
    return image, span.get('upstream_image') or image


def occurrence_flags(*owners, grant_field):
    return dict(image_loss_allowed=image_loss_allowed(*owners),
                generation_target=image_loss_allowed(*owners, grant_field=grant_field),
                non_cfg=next((o['non_cfg'] for o in reversed(owners) if o.get('non_cfg') is not None), None))
