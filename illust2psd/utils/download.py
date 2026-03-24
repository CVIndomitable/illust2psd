"""Model weight downloader with cache management."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from illust2psd.config import MODEL_CACHE_DIR

MODELS = {
    "isnet_anime": {
        "url": "https://huggingface.co/skytnt/anime-seg/resolve/main/isnetis.onnx",
        "filename": "isnetis.onnx",
        "size_mb": 176,
        "description": "ISNet anime segmentation model (foreground extraction)",
    },
    "sam2_hiera_large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "filename": "sam2.1_hiera_large.pt",
        "size_mb": 898,
        "description": "SAM2.1 Hiera Large checkpoint (body part segmentation)",
    },
    "lama": {
        "url": "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt",
        "filename": "big-lama.pt",
        "size_mb": 200,
        "description": "LaMa inpainting model",
    },
}

# Proxy support: set http(s)_proxy if port 7897 is available
_PROXY_PORT = 7897
_PROXY_URL = f"http://127.0.0.1:{_PROXY_PORT}"


class _DownloadProgressBar(tqdm):
    """tqdm wrapper for urlretrieve reporthook."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _build_opener() -> urllib.request.OpenerDirector:
    """Build URL opener, with proxy if configured."""
    proxy_env = os.environ.get("https_proxy") or os.environ.get("http_proxy")
    if proxy_env:
        proxy_handler = urllib.request.ProxyHandler({
            "http": proxy_env,
            "https": proxy_env,
        })
        return urllib.request.build_opener(proxy_handler)

    # Try local proxy on port 7897
    import socket
    try:
        sock = socket.create_connection(("127.0.0.1", _PROXY_PORT), timeout=1)
        sock.close()
        logger.info(f"Using local proxy at {_PROXY_URL}")
        proxy_handler = urllib.request.ProxyHandler({
            "http": _PROXY_URL,
            "https": _PROXY_URL,
        })
        return urllib.request.build_opener(proxy_handler)
    except (ConnectionRefusedError, OSError, socket.timeout):
        pass

    return urllib.request.build_opener()


def _download_url(url: str, dest: Path, desc: str) -> None:
    """Download a URL to a file with progress bar and proxy support."""
    opener = _build_opener()
    urllib.request.install_opener(opener)

    with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)


def get_model_path(model_name: str) -> Path:
    """Get the cached path for a model, downloading if necessary."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    info = MODELS[model_name]
    path = MODEL_CACHE_DIR / info["filename"]

    if path.exists():
        logger.debug(f"Model {model_name} found at {path}")
        return path

    logger.info(f"Downloading {model_name} ({info['size_mb']}MB)...")
    download_model(model_name)
    return path


def download_model(model_name: str, force: bool = False) -> Path:
    """Download a model to the cache directory."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    info = MODELS[model_name]
    path = MODEL_CACHE_DIR / info["filename"]

    if path.exists() and not force:
        logger.info(f"Model {model_name} already cached at {path}")
        return path

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    url = info["url"]
    logger.info(f"Downloading {model_name} from {url}")

    _download_url(url, path, info["filename"])

    logger.info(f"Saved to {path}")
    return path


def list_models() -> dict[str, dict]:
    """List all available models and their cache status."""
    result = {}
    for name, info in MODELS.items():
        path = MODEL_CACHE_DIR / info["filename"]
        result[name] = {
            **info,
            "cached": path.exists(),
            "path": str(path),
        }
    return result
