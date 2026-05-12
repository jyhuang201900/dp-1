"""Allow running the CLI as ``python -m forestseg``."""

from __future__ import annotations

from .pipeline.cli import main

if __name__ == "__main__":
    main()
