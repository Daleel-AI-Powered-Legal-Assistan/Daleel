"""
Step 5: Start Qdrant — Docker first, local file-based fallback.

If Docker is not running, uses qdrant-client's built-in local storage
(no Docker required — data saved to output/qdrant_local/).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

CONTAINER_NAME = "qdrant_labor_law"
HEALTH_URL     = "http://localhost:6333/healthz"
LOCAL_PATH     = Path("output/qdrant_db")


def _docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _container_state() -> str | None:
    res = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.State}}"],
        capture_output=True, text=True, check=False,
    )
    return res.stdout.strip() or None


def _wait_healthy(timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(HEALTH_URL, timeout=2) as r:
                if r.status == 200:
                    return True
        except (URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    return False


def _start_docker() -> bool:
    state = _container_state()
    if state == "running":
        print(f"[step5] Container {CONTAINER_NAME} already running.")
        return _wait_healthy(5)

    if state in ("exited", "created"):
        print(f"[step5] Starting existing container {CONTAINER_NAME}...")
        subprocess.run(["docker", "start", CONTAINER_NAME], check=True,
                       capture_output=True)
    else:
        storage = Path("output/qdrant_storage").resolve()
        storage.mkdir(parents=True, exist_ok=True)
        mount = f"{storage.as_posix()}:/qdrant/storage"
        print(f"[step5] Launching Qdrant container ({CONTAINER_NAME})...")
        subprocess.run(
            ["docker", "run", "-d", "--name", CONTAINER_NAME,
             "-p", "6333:6333", "-p", "6334:6334", "-v", mount, "qdrant/qdrant"],
            check=True, capture_output=True,
        )

    time.sleep(5)
    if not _wait_healthy(30):
        raise RuntimeError("Qdrant did not become healthy on http://localhost:6333")
    print("[step5] Qdrant (Docker) running at http://localhost:6333")
    return True


def run(force: bool = False) -> bool:
    if _docker_available():
        try:
            return _start_docker()
        except Exception as e:
            print(f"[step5] Docker failed ({e}), falling back to local Qdrant.")

    # Fallback: local file-based Qdrant (no Docker needed)
    LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    print(f"[step5] Using local Qdrant storage at: {LOCAL_PATH}")
    print("[step5] (Docker not available — data persisted locally)")
    return True


# ─── helper used by embed_and_index, chatbot, app ────────────────
def get_qdrant_client(collection: str | None = None):
    """
    Returns a QdrantClient connected to Docker Qdrant if available,
    otherwise connects to the local file-based storage.
    """
    from qdrant_client import QdrantClient
    # Try Docker first
    try:
        client = QdrantClient(url="http://localhost:6333", timeout=2)
        client.get_collections()
        return client
    except Exception:
        pass
    # Fall back to local
    LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(LOCAL_PATH))


if __name__ == "__main__":
    run(force="--force" in sys.argv)
