"""Convert selected source Z2 weights, or export a v3 sharded EMA checkpoint."""
import argparse
import json
from pathlib import Path

from configs.config import DEFAULTS, load_config, merge_config
from training.train import load_model
from utils.checkpoint import export_weights
from utils.weights import SOURCE_COMMIT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Local source HF directory or v3 weights.pt")
    source.add_argument("--ema-checkpoint", help="The ema directory in a v3 checkpoint")
    parser.add_argument("--config", help="v3 YAML; required if source tensors have no architecture metadata")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config) if args.config else merge_config(DEFAULTS, {})
    if args.ema_checkpoint:
        if not args.config:
            parser.error("EMA export needs --config from the completed training run")
        export_weights(args.output, ema_checkpoint=args.ema_checkpoint, config=config)
    else:
        config["model"]["checkpoint"] = args.input
        model = load_model(config)
        export_weights(args.output, model=model, config=config)
    Path(str(args.output) + ".json").write_text(json.dumps({
        "schema": "umm_v3_conversion_v1", "reference_commit": SOURCE_COMMIT,
        "input": args.input or args.ema_checkpoint, "output": args.output,
        "training_state_migrated": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
