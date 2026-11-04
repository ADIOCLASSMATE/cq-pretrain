"""Supported preprocessing entrypoint; legacy writers are never the fallback."""
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "joint":
        from .unified.__main__ import main
        raise SystemExit(main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "process":
        raise SystemExit('Retired entrypoint: use python -m corpus_pipeline prepare/run --config ...; existing runs use their frozen runtime.')
    from .orchestration.__main__ import main
    raise SystemExit(main(sys.argv[1:]))
