"""CLI entrypoint: python -m chest_xray_classifier"""
from __future__ import annotations

import sys


def main() -> int:
    print("chest-xray-classifier — use make train / make evaluate / make serve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
