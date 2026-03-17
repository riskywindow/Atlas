"""Host-native mock worker for Compose smoke checks."""

from __future__ import annotations

import uvicorn

from switchyard.adapters.mock import MockBackendAdapter
from switchyard.worker.app import create_worker_app


def main() -> None:
    """Run a lightweight host-native mock worker on macOS or any local dev machine."""

    app = create_worker_app(
        MockBackendAdapter(name="mock-host-worker"),
        worker_name="mock-host-worker",
        warmup_on_start=True,
    )
    uvicorn.run(app, host="127.0.0.1", port=8101, log_level="info")


if __name__ == "__main__":
    main()
