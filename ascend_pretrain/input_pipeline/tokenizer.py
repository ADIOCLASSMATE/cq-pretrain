"""Keep online token IDs tied to the loaded language checkpoint, not corpus counts."""
import json
from pathlib import Path


def tokenizer_root(root,cfg):
    if cfg is not None and cfg.text_assets:
        from ..qwen35 import asset_path
        return asset_path(cfg.text_assets)
    return Path(root)/'tokenizer'


def tokenizer_metadata(cfg,published):
    if cfg is None or not cfg.text_assets:
        return published
    from ..qwen35 import asset_manifest,asset_path
    from ..prepare_qwen35 import sha256
    root=asset_path(cfg.text_assets);manifest=asset_manifest(root)
    files={name:manifest['files'][name] for name in ('tokenizer.json','tokenizer_config.json')}
    for name,digest in files.items():
        if sha256(root/name)!=digest:raise ValueError(f'Online tokenizer checksum changed: {name}')
    return dict(model_id=manifest['model_id'],revision=manifest['revision'],files=files,
                counting=dict(exclude_special_token_ids=sorted(special_token_ids(root))))


def special_token_ids(root):
    config=json.loads((Path(root)/'tokenizer_config.json').read_text())
    return {int(k) for k,v in config['added_tokens_decoder'].items() if v.get('special',False)}
