"""Explicit unresolved-image views retain ordered originals outside training."""
from corpus_pipeline.production.common import canonical, digest
from corpus_pipeline.dedup.training_release import text_fingerprint


def row(unit, segments, locator, direction, source_key, reason, annotation):
    locator = {**locator, 'adapter': 'genund.loose_original_supervision.v1', 'source': unit['source'],
               'view_direction': direction, 'processing_policy': 'corpus.task_policy.v1',
               'dataset_id': unit['dataset_id']}
    original = {'sample_id': digest({'locator': locator, 'segments': segments}), 'source': unit['source'],
                'source_shard': unit['source_shard'], 'source_key': source_key,
                'format': 'conversation' if unit['source'] == 'pixmo-ask-model-anything' else 'caption',
                'direction': direction, 'processing_policy': 'corpus.task_policy.v1',
                'image_loss_allowed': False, 'image_count': 1, 'segments': segments,
                'source_annotation_locator': locator, 'source_annotation': annotation}
    # Never pretend the missing image exists. The full ordered sequence remains
    # in original_supervision_json; the outer view is mandatory quarantine.
    visible = [dict(segment, image_index=None) for segment in segments if segment['kind'] == 'text']
    metadata = {'adapter': locator['adapter'], 'direction': direction, 'processing_policy': 'corpus.task_policy.v1',
                'image_loss_allowed': False, 'pairing_state': 'unresolved',
                'prepared_adapter_status': 'quarantine', 'binding_contract': 'unresolved_original_image_binding.v1',
                'issues': ['unresolved_original_image_binding'], 'binding_failure_reason': reason,
                'actual_image_members': 0, 'original_image_tokens': 1, 'binding_repair_applied': False}
    return {'collection': 'genund', **{key: original[key] for key in ('source','sample_id','source_shard','source_key','format')},
            'segments': visible, 'images': [], 'image_count': 0, 'text_hash': text_fingerprint(visible),
            'original_supervision_json': canonical(original), 'original_locator_json': canonical(locator),
            'adapter_metadata_json': canonical(metadata)}
