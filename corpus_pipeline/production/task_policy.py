"""Explicit task routing for new runs; immutable legacy releases keep their policy."""
from copy import deepcopy

VERSION = 'corpus.task_policy.v1'
PROFILE = 'task_aware_v1'


def enabled(rules):
    value = rules.get('task_routing')
    if value not in (None, VERSION):
        raise ValueError('Unsupported task routing policy: ' + str(value))
    return value == VERSION


def task_route(record):
    category = record.get('category')
    if category in ('single_i2t', 'multi_i2t'):
        return 'i2t'
    if category in ('single_t2i', 'multi_t2i'):
        return 't2i'
    if category == 'pure_text':
        return 'text'
    # Interleaving is a layout, not a generation direction. Only explicit
    # task metadata or uniformly user-side image slots prove an I2T view.
    images = [s for s in record.get('segments', []) if s.get('kind') == 'image']
    roles = {s.get('role') for s in images}
    direction = record.get('metadata', {}).get('direction')
    if images and roles == {'user'}:
        return 'i2t'
    if images and direction in ('i2t', 'image_to_text') and 'assistant' not in roles:
        return 'i2t'
    if images and roles == {'assistant'}:
        return 't2i'
    if images and direction in ('t2i', 'text_to_image') and 'user' not in roles:
        return 't2i'
    if category in ('interleaved', 'mixed_multimodal'):
        return 'mixed'
    return 'unknown'


def fields(record):
    route = task_route(record)
    allowed = route == 't2i'
    for key in ('image_loss_allowed', 'dcvlm_image_loss_allowed'):
        value = record.get('metadata', {}).get(key)
        if value is not None:
            if type(value) is not bool:
                raise ValueError(key + ' must be boolean')
            allowed = allowed and value
    return {'processing_policy': VERSION, 'task_route': route,
            'image_loss_allowed': allowed}


def run_rules(cfg, legacy):
    rules = deepcopy(legacy)
    if cfg.get('policy', {}).get('profile') == PROFILE:
        rules['task_routing'] = VERSION
        rules['version'] = VERSION
        policy = receipt_policy(cfg)
        if 't2i_uncertain' in policy:
            rules['t2i_uncertain'] = policy['t2i_uncertain']
    return rules


def receipt_policy(cfg):
    # The rest of the execution config and source rules are independently
    # pinned in EXECUTION.json and the frontpass plan.
    value = cfg.get('policy', {})
    mode = value.get('t2i_uncertain', 'retain_annotated')
    if value.get('profile') != PROFILE:
        if 't2i_uncertain' in value:
            raise ValueError('t2i_uncertain requires task_aware_v1')
        return None
    if mode not in ('retain_annotated', 'exclude'):
        raise ValueError('Unsupported t2i_uncertain policy')
    result = {'task_routing': VERSION}
    if mode == 'exclude':
        result['t2i_uncertain'] = mode
    return result
