#!/usr/bin/env python3
"""One-click model weight download script."""

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from illust2psd.utils.download import MODELS, download_model


def main():
    print("illust2psd Model Downloader")
    print("=" * 40)

    for name, info in MODELS.items():
        print(f"\n{name}: {info['description']} ({info['size_mb']}MB)")

    print("\nDownloading all models...")
    for name in MODELS:
        try:
            download_model(name)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
